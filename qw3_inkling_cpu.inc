/* Serial graph for the original Inkling 2 Mini text GGUF.
 * The mapped weights are indexed by physical layer and mutable state by visit. */
enum {
    INK_D = 2304, INK_FF = 1152, INK_HEADS = 24, INK_KV_HEADS = 8,
    INK_H = 128, INK_Q = INK_HEADS * INK_H, INK_KV = INK_KV_HEADS * INK_H,
    INK_REL = 16, INK_RANK = 64, INK_EXPERTS = 128, INK_TOP = 8,
    INK_CONV = 4, INK_CHANNELS = 2 * INK_KV + 2 * INK_D,
    INK_RES_BLOCKS = INKLING_LAYERS / (INKLING_CORE_START),
};

typedef struct {
    const qw3_tensor *q, *k, *v, *gate, *wo, *relative, *bias, *sink;
    const qw3_tensor *q_clip, *rel_clip, *q_norm, *k_norm, *v_norm, *attn_res;
    const qw3_tensor *k_conv, *v_conv, *attn_conv, *ffn_conv, *k_taps, *v_taps;
    const qw3_tensor *ffn_res, *ffn_norm, *router, *router_bias;
    const qw3_tensor *gate_up, *down, *gate_up_scale, *down_scale;
    const qw3_tensor *loop_down[4], *loop_gain[4], *loop_forget[4];
} inkling_layer;

struct qw3_inkling_weights {
    inkling_layer layer[INKLING_LAYERS];
    const qw3_tensor *embedding, *embedding_norm, *output, *output_norm, *output_res;
    const qw3_tensor *loop_res, *loop_kv_res;
};

typedef struct {
    float *k, *v;
    unsigned capacity;
    float conv[INK_CONV - 1][INK_CHANNELS];
} inkling_visit_state;

struct qw3_inkling_state {
#ifndef QW3_NO_METAL
    qw3_metal_inkling_session *metal;
#endif
    inkling_visit_state *visit;
};

static const qw3_tensor *inkling_tensor(const qw3_model *m, unsigned layer, const char *suffix) {
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return model_find_tensor(m, name);
}

static qw3_inkling_weights *inkling_bind(const qw3_model *m) {
    qw3_inkling_weights *w = qw3_xcalloc(1, sizeof(*w));
    w->embedding = model_find_tensor(m, "token_embd.weight");
    w->embedding_norm = model_find_tensor(m, "token_embd_norm.weight");
    w->output = model_find_tensor(m, "output.weight");
    w->output_norm = model_find_tensor(m, "output_norm.weight");
    w->output_res = model_find_tensor(m, "output_res_query.weight");
    w->loop_res = model_find_tensor(m, "loop_res_query.weight");
    w->loop_kv_res = model_find_tensor(m, "loop_kv_res_query.weight");
    for (unsigned i = 0; i < INKLING_LAYERS; i++) {
        inkling_layer *l = &w->layer[i];
#define T(field, name) l->field = inkling_tensor(m, i, name)
        T(q, "attn_q.weight"); T(k, "attn_k.weight"); T(v, "attn_v.weight");
        T(gate, "attn_gate.weight"); T(wo, "attn_output.weight");
        T(relative, "attn_rel.weight"); T(bias, "attn_rel_bias.weight"); T(sink, "attn_sink_proj.weight");
        T(q_clip, "attn_q_clip.weight"); T(rel_clip, "attn_rel_clip.weight");
        T(attn_res, "attn_res_query.weight");
        T(q_norm, "attn_norm.weight");
        if (l->q_norm) {
            l->k_norm = l->v_norm = l->q_norm;
        } else {
            T(q_norm, "attn_q_inp_norm.weight");
            T(k_norm, "attn_k_inp_norm.weight"); T(v_norm, "attn_v_inp_norm.weight");
            T(k_taps, "attn_k_conv_proj.weight"); T(v_taps, "attn_v_conv_proj.weight");
        }
        T(k_conv, "attn_k_conv.weight"); T(v_conv, "attn_v_conv.weight");
        T(attn_conv, "attn_post_conv.weight"); T(ffn_conv, "ffn_post_conv.weight");
        T(ffn_res, "ffn_res_query.weight"); T(ffn_norm, "ffn_norm.weight");
        T(router, "ffn_gate_inp.weight"); T(router_bias, "exp_probs_b.bias");
        T(gate_up, "ffn_gate_up_exps.weight"); T(gate_up_scale, "ffn_gate_up_exps.scale");
        T(down, "ffn_down_exps.weight"); T(down_scale, "ffn_down_exps.scale");
#undef T
        if (i >= INKLING_CORE_START && i < INKLING_CORE_END) {
            const char *modules[] = {"attn", "k", "v", "mlp"};
            for (unsigned j = 0; j < 4; j++) {
                char name[48];
                snprintf(name, sizeof(name), "loop_%s_down.weight", modules[j]);
                l->loop_down[j] = inkling_tensor(m, i, name);
                snprintf(name, sizeof(name), "loop_%s_gain.weight", modules[j]);
                l->loop_gain[j] = inkling_tensor(m, i, name);
                snprintf(name, sizeof(name), "loop_%s_forget.weight", modules[j]);
                l->loop_forget[j] = inkling_tensor(m, i, name);
            }
        }
    }
    return w;
}

static unsigned inkling_visit_capacity(const qw3_model *m, unsigned visit, unsigned ctx) {
    const unsigned window = (unsigned)m->inkling.window;
    return m->inkling.sliding[visit] && ctx > window ? window : ctx;
}

static qw3_context_memory inkling_context_memory(const qw3_model *m, int ctx_size, bool metal) {
    qw3_context_memory mem = {0};
    if (ctx_size <= 0 || ctx_size > m->meta.context_length) return mem;
    for (unsigned i = 0; i < INKLING_VISITS; i++) {
        mem.gqa_kv_bytes += (uint64_t)inkling_visit_capacity(m, i, (unsigned)ctx_size) *
                           2 * INK_KV * (metal ? sizeof(uint16_t) : sizeof(float));
    }
    mem.scratch_bytes = sizeof(qw3_inkling_state) +
                        (uint64_t)m->inkling.vocab_unpadded * sizeof(float);
    if (metal) {
        const uint64_t partitions = ((uint64_t)ctx_size + 127) / 128;
        mem.scratch_bytes += 24 * INK_D * sizeof(float) +
                             INK_TOP * (2 * INK_FF + INK_D) * sizeof(float) +
                             INK_HEADS * partitions * (INK_H + 4) * sizeof(float) +
                             (uint64_t)m->inkling.vocab_unpadded * sizeof(float);
        mem.deltanet_state_bytes = (uint64_t)INKLING_VISITS * (INK_CONV - 1) * INK_CHANNELS * sizeof(float);
    } else {
        mem.deltanet_state_bytes = INKLING_VISITS * sizeof(inkling_visit_state);
    }
    mem.total_bytes = mem.gqa_kv_bytes + mem.scratch_bytes + mem.deltanet_state_bytes;
    return mem;
}

static qw3_inkling_state *inkling_state_create(const qw3_model *m, unsigned ctx, bool metal) {
    qw3_inkling_state *s = qw3_xcalloc(1, sizeof(*s));
#ifndef QW3_NO_METAL
    if (metal) {
        uint32_t capacities[INKLING_VISITS];
        for (unsigned i = 0; i < INKLING_VISITS; i++) {
            capacities[i] = inkling_visit_capacity(m, i, ctx);
        }
        s->metal = qw3_metal_inkling_session_create(capacities, INKLING_VISITS);
        if (!s->metal) {
            free(s);
            return NULL;
        }
        return s;
    }
#else
    (void)metal;
#endif
    s->visit = qw3_xcalloc(INKLING_VISITS, sizeof(*s->visit));
    for (unsigned i = 0; i < INKLING_VISITS; i++) {
        inkling_visit_state *v = &s->visit[i];
        v->capacity = inkling_visit_capacity(m, i, ctx);
        v->k = qw3_xcalloc((size_t)v->capacity * INK_KV, sizeof(float));
        v->v = qw3_xcalloc((size_t)v->capacity * INK_KV, sizeof(float));
    }
    return s;
}

static void inkling_state_reset(qw3_inkling_state *s) {
    // A replay always starts at position zero and replaces every visible KV row.
    // The Metal convolutions treat the first token as having zero history.
    if (!s->visit) return;
    for (unsigned i = 0; i < INKLING_VISITS; i++) {
        memset(s->visit[i].conv, 0, sizeof(s->visit[i].conv));
    }
}

static void inkling_state_free(qw3_inkling_state *s) {
    if (!s) return;
#ifndef QW3_NO_METAL
    qw3_metal_inkling_session_free(s->metal);
#endif
    if (s->visit) {
        for (unsigned i = 0; i < INKLING_VISITS; i++) {
            free(s->visit[i].k);
            free(s->visit[i].v);
        }
        free(s->visit);
    }
    free(s);
}

static const float *inkling_f32(const qw3_model *m, const qw3_tensor *w) {
    return (const float *)tensor_data(m, w);
}

static float inkling_dot(const float *a, const float *b, unsigned n) {
    float sum = 0.0f;
    for (unsigned i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float inkling_inv_rms(const float *a, unsigned n) {
    return 1.0f / sqrtf(inkling_dot(a, a, n) / n + QW3_RMS_EPS);
}

static float inkling_attention_scale(const qw3_model *m, unsigned pos, bool sliding) {
    return sliding ? 1.0f :
        1.0f + m->inkling.log_alpha * logf(fmaxf(1.0f, (float)(pos + 1) / m->inkling.log_floor));
}

static void inkling_mix(float *out, const float *cur,
                         const float banks[][INK_D], unsigned n, const float *query) {
    float scales[INK_RES_BLOCKS + 1], scores[INK_RES_BLOCKS + 1];
    scales[n] = inkling_inv_rms(cur, INK_D);
    scores[n] = inkling_dot(cur, query, INK_D) * scales[n] / sqrtf(INK_D);
    float max = scores[n];
    for (unsigned i = 0; i < n; i++) {
        scales[i] = inkling_inv_rms(banks[i], INK_D);
        scores[i] = inkling_dot(banks[i], query, INK_D) * scales[i] / sqrtf(INK_D);
        if (scores[i] > max) max = scores[i];
    }
    float norm = 0.0f;
    for (unsigned i = 0; i <= n; i++) {
        scores[i] = expf(scores[i] - max);
        norm += scores[i];
    }
    for (unsigned j = 0; j < INK_D; j++) {
        float sum = cur[j] * scales[n] * scores[n];
        for (unsigned i = 0; i < n; i++) sum += banks[i][j] * scales[i] * scores[i];
        out[j] = sum / norm;
    }
}

typedef struct {
    const qw3_tensor *tensor;
    const float *input;
    float *output;
    unsigned expert;
} inkling_projection;

static qw3_tensor inkling_projection_slice(const inkling_projection *p) {
    qw3_tensor w = *p->tensor;
    if (w.ndim == 3) {
        w.ndim = 2;
        w.offset += (uint64_t)p->expert * w.dim[1] * inkling_row_bytes(w.type, w.dim[0]);
    }
    return w;
}

static bool inkling_project(qw3_session *s, const inkling_projection *projections, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        const inkling_projection *p = &projections[i];
        const qw3_tensor w = inkling_projection_slice(p);
        if (!cpu_matvec(&s->engine->model, &w, p->input, p->output)) return false;
    }
    return true;
}

static bool inkling_matvec(qw3_session *s, const qw3_tensor *w, const float *input, float *out) {
    const inkling_projection projection = {w, input, out, 0};
    return inkling_project(s, &projection, 1);
}

static bool inkling_control(qw3_session *s, const inkling_layer *l,
                            unsigned first, unsigned count, float cur[][INK_D],
                            const float message[][INK_D]) {
    float input[3][2 * INK_D];
    for (unsigned j = 0; j < count; j++) {
        const float ic = inkling_inv_rms(cur[j], INK_D);
        const float im = inkling_inv_rms(message[j], INK_D);
        for (unsigned i = 0; i < INK_D; i++) {
            input[j][i] = cur[j][i] * ic;
            input[j][INK_D + i] = message[j][i] * im;
        }
    }
    float code[3][INK_RANK], gain[3][INK_D], forget[3][INK_D];
    inkling_projection down[3], project[6];
    for (unsigned j = 0; j < count; j++) {
        down[j] = (inkling_projection){l->loop_down[first + j], input[j], code[j], 0};
        project[2 * j] = (inkling_projection){l->loop_gain[first + j], code[j], gain[j], 0};
        project[2 * j + 1] = (inkling_projection){l->loop_forget[first + j], code[j], forget[j], 0};
    }
    if (!inkling_project(s, down, count) || !inkling_project(s, project, 2 * count)) return false;
    for (unsigned j = 0; j < count; j++) {
        for (unsigned i = 0; i < INK_D; i++) {
            cur[j][i] = (1.0f - tanhf(forget[j][i])) * cur[j][i] +
                        tanhf(gain[j][i]) * message[j][i];
        }
    }
    return true;
}

static void inkling_conv(float *x, unsigned n, const float *w, const float *taps,
                          inkling_visit_state *s, unsigned offset) {
    const float scale = 1.0f / sqrtf((taps ? 2.0f : 1.0f) * INK_CONV + 1.0f);
    for (unsigned i = 0; i < n; i++) {
        const unsigned j = offset + i;
        const float a = s->conv[0][j], b = s->conv[1][j], c = s->conv[2][j], d = x[i];
        const float *weight = w + i * INK_CONV;
        float value = a * weight[0] + b * weight[1] + c * weight[2] + d * weight[3];
        if (taps) {
            const float *t = taps + (i / INK_H) * INK_CONV;
            value += d * t[0] + c * t[1] + b * t[2] + a * t[3];
        }
        s->conv[0][j] = b;
        s->conv[1][j] = c;
        s->conv[2][j] = d;
        x[i] = (d + value) * scale;
    }
}

static void inkling_clip(float *x, unsigned count, unsigned width, const float *weight, float limit) {
    for (unsigned h = 0; h < count; h++) {
        float *row = x + h * width;
        if (weight) {
            for (unsigned i = 0; i < width; i++) row[i] *= weight[h * width + i];
        }
        const float scale = fminf(1.0f, limit / (sqrtf(inkling_dot(row, row, width)) + 1.0e-6f));
        for (unsigned i = 0; i < width; i++) row[i] *= scale;
    }
}

static bool inkling_attention(qw3_session *s, const inkling_layer *l,
                               unsigned visit_index, unsigned pos, bool sliding,
                               float input[3][INK_D], float *out) {
    const qw3_model *m = &s->engine->model;
    const float tau = inkling_attention_scale(m, pos, sliding);
    inkling_visit_state *visit = &s->inkling->visit[visit_index];
    float q[INK_Q], k[INK_KV], v[INK_KV], gate[INK_Q], mixed[INK_Q];
    float rel[INK_HEADS * INK_REL], sinks[INK_HEADS], kt[INK_KV_HEADS * INK_CONV];
    float vt[INK_KV_HEADS * INK_CONV];
    const inkling_projection projections[] = {
        {l->q, input[0], q, 0}, {l->k, input[1], k, 0},
        {l->v, input[2], v, 0}, {l->gate, input[0], gate, 0},
        {l->relative, input[0], rel, 0}, {l->sink, input[0], sinks, 0},
        {l->k_taps, input[1], kt, 0}, {l->v_taps, input[2], vt, 0},
    };
    if (!inkling_project(s, projections, sliding ? 6 : 8)) return false;
    inkling_conv(k, INK_KV, inkling_f32(m, l->k_conv), sliding ? NULL : kt, visit, 0);
    inkling_conv(v, INK_KV, inkling_f32(m, l->v_conv), sliding ? NULL : vt, visit, INK_KV);
    inkling_clip(q, INK_HEADS, INK_H, inkling_f32(m, l->q_clip), m->inkling.qk_clip);
    inkling_clip(k, INK_KV_HEADS, INK_H, NULL, m->inkling.qk_clip);
    inkling_clip(rel, INK_HEADS, INK_REL, inkling_f32(m, l->rel_clip), m->inkling.rel_clip);
    for (unsigned h = 0; h < INK_KV_HEADS; h++) {
        float *row = v + h * INK_H;
        const float scale = inkling_inv_rms(row, INK_H);
        for (unsigned i = 0; i < INK_H; i++) row[i] *= scale;
    }
    const unsigned slot = pos % visit->capacity;
    memcpy(visit->k + (size_t)slot * INK_KV, k, sizeof(k));
    memcpy(visit->v + (size_t)slot * INK_KV, v, sizeof(v));
    const unsigned first = pos + 1 > visit->capacity ? pos + 1 - visit->capacity : 0;
    const float *bias = inkling_f32(m, l->bias);
    for (unsigned h = 0; h < INK_HEADS; h++) {
        const unsigned kv_head = h / (INK_HEADS / INK_KV_HEADS);
        float *dst = mixed + h * INK_H;
        memset(dst, 0, INK_H * sizeof(float));
        float max = sinks[h], sum = 1.0f; // Sink is one extra logit with zero value.
        for (unsigned j = first; j <= pos; j++) {
            const size_t offset = (size_t)(j % visit->capacity) * INK_KV + kv_head * INK_H;
            const unsigned distance = pos - j;
            float score = inkling_dot(q + h * INK_H, visit->k + offset, INK_H) / sqrtf(INK_H);
            if (distance < (unsigned)m->inkling.relative_extent) {
                score += inkling_dot(rel + h * INK_REL, bias + distance * INK_REL, INK_REL);
            }
            score *= tau;
            const float next = fmaxf(max, score);
            const float keep = expf(max - next), add = expf(score - next);
            sum = sum * keep + add;
            for (unsigned i = 0; i < INK_H; i++) dst[i] = dst[i] * keep + visit->v[offset + i] * add;
            max = next;
        }
        for (unsigned i = 0; i < INK_H; i++) dst[i] /= sum * (1.0f + expf(-gate[h * INK_H + i]));
    }
    if (!inkling_matvec(s, l->wo, mixed, out)) return false;
    inkling_conv(out, INK_D, inkling_f32(m, l->attn_conv), NULL, visit, 2 * INK_KV);
    return true;
}

static bool inkling_ffn(qw3_session *s, const inkling_layer *l, const float *x, float *out) {
    const qw3_model *m = &s->engine->model;
    float router[INK_EXPERTS], choices[INK_EXPERTS], selected[INK_TOP];
    int ids[INK_TOP];
    if (!inkling_matvec(s, l->router, x, router)) return false;
    const float *bias = l->router_bias ? inkling_f32(m, l->router_bias) : NULL;
    const float *gate_scale = l->gate_up_scale ? inkling_f32(m, l->gate_up_scale) : NULL;
    const float *down_scale = l->down_scale ? inkling_f32(m, l->down_scale) : NULL;
    for (unsigned i = 0; i < INK_EXPERTS; i++) {
        router[i] = 1.0f / (1.0f + expf(-router[i]));
        choices[i] = router[i] + (bias ? bias[i] : 0.0f);
    }
    topk_desc(choices, INK_EXPERTS, INK_TOP, ids, selected);
    float sum = 1.0e-20f;
    for (unsigned i = 0; i < INK_TOP; i++) sum += router[ids[i]];
    float weights[INK_TOP], scales[INK_TOP];
    for (unsigned i = 0; i < INK_TOP; i++) {
        const int id = ids[i];
        weights[i] = router[id] / sum * m->inkling.expert_scale * (down_scale ? down_scale[id] : 1.0f);
        scales[i] = gate_scale ? gate_scale[id] : 1.0f;
    }
    const float beta = m->inkling.situ_beta, limit = m->inkling.situ_up;
    float gate_up[INK_TOP][2 * INK_FF], down[INK_TOP][INK_D];
    inkling_projection projections[INK_TOP];
    for (unsigned i = 0; i < INK_TOP; i++) {
        projections[i] = (inkling_projection){l->gate_up, x, gate_up[i], (unsigned)ids[i]};
    }
    if (!inkling_project(s, projections, INK_TOP)) return false;
    for (unsigned i = 0; i < INK_TOP; i++) {
        const int id = ids[i];
        const float scale = scales[i];
        for (unsigned j = 0; j < INK_FF; j++) {
            const float gate = gate_up[i][j] * scale, up = gate_up[i][INK_FF + j] * scale;
            gate_up[i][j] = beta * tanhf(gate / beta) / (1.0f + expf(-gate)) *
                            limit * tanhf(up / limit);
        }
        projections[i] = (inkling_projection){l->down, gate_up[i], down[i], (unsigned)id};
    }
    if (!inkling_project(s, projections, INK_TOP)) return false;
    memset(out, 0, INK_D * sizeof(float));
    for (unsigned i = 0; i < INK_TOP; i++) {
        for (unsigned j = 0; j < INK_D; j++) out[j] += weights[i] * down[i][j];
    }
    return true;
}

#ifndef QW3_NO_METAL
enum {
    INK_CUR, INK_OUT, INK_INPUT,
    INK_MESSAGE = INK_INPUT + 3,
    INK_BLOCKS = INK_MESSAGE + 4,
    INK_PRELUDE = INK_BLOCKS + INK_RES_BLOCKS,
    INK_BANKS,
    INK_BANK_LAST = INK_BANKS + INK_RES_BLOCKS,
};

static qw3_metal_projection inkling_metal_projection(const qw3_tensor *tensor, unsigned expert) {
    const uint64_t row_bytes = inkling_row_bytes(tensor->type, tensor->dim[0]);
    return (qw3_metal_projection){
        .tensor_offset = tensor->offset + expert * tensor->dim[1] * row_bytes,
        .type = tensor->type, .n_in = (uint32_t)tensor->dim[0], .n_out = (uint32_t)tensor->dim[1],
        .row_bytes = (uint32_t)row_bytes
    };
}

typedef struct {
    qw3_metal_inkling_session *gpu;
    unsigned count;
} inkling_metal_graph;

static bool inkling_metal_vector_op(inkling_metal_graph graph,
        unsigned dst, unsigned src, unsigned count, bool add) {
    return graph.count > 1
        ? qw3_metal_inkling_batch_vector_op(graph.gpu, dst, src, count, graph.count, add)
        : qw3_metal_inkling_vector_op(graph.gpu, dst, src, count, add);
}

static bool inkling_metal_mix(inkling_metal_graph graph,
        unsigned dst, unsigned src, unsigned history, unsigned banks, uint64_t query) {
    return graph.count > 1
        ? qw3_metal_inkling_batch_mix(graph.gpu, dst, src, history, banks, query, graph.count)
        : qw3_metal_inkling_mix(graph.gpu, dst, src, history, banks, query);
}

static bool inkling_metal_mix_norm(inkling_metal_graph graph,
        unsigned dst, unsigned src, unsigned history, unsigned banks,
        uint64_t query, uint64_t norm_weights) {
    return graph.count > 1
        ? qw3_metal_inkling_batch_mix_norm(graph.gpu, dst, src, history, banks, query, norm_weights, graph.count)
        : qw3_metal_inkling_mix_norm(graph.gpu, dst, src, history, banks, query, norm_weights);
}

static bool inkling_metal_mix4(inkling_metal_graph graph,
        unsigned dst_base, unsigned src, unsigned history, unsigned banks,
        uint64_t q_attn, uint64_t q_kv) {
    return graph.count > 1
        ? qw3_metal_inkling_batch_mix4(graph.gpu, dst_base, src, history, banks, q_attn, q_kv, graph.count)
        : qw3_metal_inkling_mix4(graph.gpu, dst_base, src, history, banks, q_attn, q_kv);
}

static bool inkling_metal_norm(inkling_metal_graph graph, unsigned vector, uint64_t weights) {
    return graph.count > 1
        ? qw3_metal_inkling_batch_norm(graph.gpu, vector, weights, graph.count)
        : qw3_metal_inkling_norm(graph.gpu, vector, weights);
}

static bool inkling_apply_metal_control(inkling_metal_graph graph,
        const inkling_layer *l, unsigned module, unsigned state) {
    const qw3_metal_inkling_control control = {
        l->loop_down[module]->offset, l->loop_gain[module]->offset, l->loop_forget[module]->offset
    };
    return graph.count > 1
        ? qw3_metal_inkling_batch_apply_control(graph.gpu, &control, state, INK_MESSAGE + module, graph.count)
        : qw3_metal_inkling_apply_control(graph.gpu, &control, state, INK_MESSAGE + module);
}

static bool inkling_apply_metal_control_from(inkling_metal_graph graph,
        const inkling_layer *l, unsigned module, unsigned src_state, unsigned dst_state) {
    const qw3_metal_inkling_control control = {
        l->loop_down[module]->offset, l->loop_gain[module]->offset, l->loop_forget[module]->offset
    };
    return graph.count > 1
        ? qw3_metal_inkling_batch_apply_control_from(graph.gpu, &control, src_state, dst_state,
                                                     INK_MESSAGE + module, graph.count)
        : qw3_metal_inkling_apply_control_from(graph.gpu, &control, src_state, dst_state,
                                               INK_MESSAGE + module);
}

static bool inkling_mapped_expert_type(uint32_t type) {
    switch (type) {
    case QW3_TENSOR_Q8_0:
    case QW3_TENSOR_Q3_K:
    case QW3_TENSOR_Q4_K:
    case QW3_TENSOR_Q5_K:
    case QW3_TENSOR_Q6_K:
    case QW3_TENSOR_IQ2_XS:
    case QW3_TENSOR_IQ3_XXS:
    case QW3_TENSOR_IQ4_NL:
    case QW3_TENSOR_IQ3_S:
    case QW3_TENSOR_IQ2_S:
    case QW3_TENSOR_IQ4_XS:
    case QW3_TENSOR_NVFP4:
        return true;
    default:
        return false;
    }
}

static bool inkling_metal_attention(inkling_metal_graph graph,
        const qw3_model *m, const inkling_layer *l, unsigned visit, unsigned pos,
        bool sliding, bool shared_qkv, unsigned dst, bool add_dst) {
    qw3_metal_inkling_attention a = {
        .q = inkling_metal_projection(l->q, 0), .k = inkling_metal_projection(l->k, 0),
        .v = inkling_metal_projection(l->v, 0), .gate = inkling_metal_projection(l->gate, 0),
        .relative = inkling_metal_projection(l->relative, 0), .sink = inkling_metal_projection(l->sink, 0),
        .output = inkling_metal_projection(l->wo, 0),
        .k_conv = l->k_conv->offset, .v_conv = l->v_conv->offset, .q_clip = l->q_clip->offset,
        .relative_clip = l->rel_clip->offset, .relative_bias = l->bias->offset,
        .post_conv = l->attn_conv->offset,
        .qk_limit = m->inkling.qk_clip, .relative_limit = m->inkling.rel_clip,
        .tau = inkling_attention_scale(m, pos, sliding),
        .relative_extent = (uint32_t)m->inkling.relative_extent, .sliding = sliding,
        .shared_qkv = shared_qkv, .add_dst = add_dst,
    };
    if (!sliding) {
        a.k_taps = inkling_metal_projection(l->k_taps, 0);
        a.v_taps = inkling_metal_projection(l->v_taps, 0);
    }
    if (graph.count > 1) {
        return qw3_metal_inkling_batch_attend(graph.gpu, &a, INK_INPUT, dst, visit, pos,
                                              graph.count, m->inkling.log_alpha, m->inkling.log_floor);
    }
    return qw3_metal_inkling_attend(graph.gpu, &a, INK_INPUT, dst, visit, pos);
}

static bool inkling_metal_ffn(inkling_metal_graph graph,
        const qw3_model *m, const inkling_layer *l, unsigned visit, unsigned pos, bool *added_cur) {
    *added_cur = false;
    const qw3_metal_inkling_router router = {
        .matrix = inkling_metal_projection(l->router, 0),
        .bias = l->router_bias ? l->router_bias->offset : 0,
        .gate_scale = l->gate_up_scale ? l->gate_up_scale->offset : 0,
        .down_scale = l->down_scale ? l->down_scale->offset : 0,
        .scale = m->inkling.expert_scale,
    };
    qw3_metal_inkling_session *gpu = graph.gpu;
    if (graph.count > 1) {
        qw3_metal_inkling_moe_weights moe = {
            .gate_up = inkling_metal_projection(l->gate_up, 0),
            .down = inkling_metal_projection(l->down, 0),
        };
        moe.gate_up_expert_bytes = moe.gate_up.n_out * moe.gate_up.row_bytes;
        moe.down_expert_bytes = moe.down.n_out * moe.down.row_bytes;
        *added_cur = true;
        return qw3_metal_inkling_batch_ffn(gpu, &router, &moe, INK_INPUT, INK_CUR,
                                          m->inkling.situ_beta, m->inkling.situ_up,
                                          visit, pos, graph.count, l->ffn_conv->offset);
    }
    if (inkling_mapped_expert_type(l->gate_up->type) &&
        inkling_mapped_expert_type(l->down->type)) {
        qw3_metal_inkling_moe_weights moe = {
            .gate_up = inkling_metal_projection(l->gate_up, 0),
            .down = inkling_metal_projection(l->down, 0),
        };
        moe.gate_up_expert_bytes = moe.gate_up.n_out * moe.gate_up.row_bytes;
        moe.down_expert_bytes = moe.down.n_out * moe.down.row_bytes;
        *added_cur = true;
        return qw3_metal_inkling_ffn(gpu, &router, &moe, INK_INPUT, INK_CUR,
                                     m->inkling.situ_beta, m->inkling.situ_up,
                                     visit, pos, l->ffn_conv->offset);
    }
    uint32_t ids[INK_TOP];
    if (!qw3_metal_inkling_route(gpu, &router, INK_INPUT, ids)) return false;
    qw3_metal_inkling_expert experts[INK_TOP];
    for (unsigned i = 0; i < INK_TOP; i++) {
        if (ids[i] >= INK_EXPERTS) return false;
        experts[i] = (qw3_metal_inkling_expert){
            inkling_metal_projection(l->gate_up, ids[i]), inkling_metal_projection(l->down, ids[i])
        };
    }
    return qw3_metal_inkling_experts(gpu, experts, INK_INPUT, INK_OUT,
                                    m->inkling.situ_beta, m->inkling.situ_up,
                                    visit, pos, l->ffn_conv->offset);
}

static bool inkling_eval_metal(qw3_session *s, const int *tokens, const float *embd,
                               unsigned count, unsigned pos, bool compute_logits) {
    const qw3_model *m = &s->engine->model;
    const qw3_inkling_weights *w = s->engine->inkling;
    qw3_metal_inkling_session *gpu = s->inkling->metal;
    const inkling_metal_graph graph = {gpu, count};
    const qw3_metal_projection embedding = inkling_metal_projection(w->embedding, 0);
    const uint64_t row_bytes = INK_D * sizeof(float);
    unsigned n_blocks = 0, n_prelude = 0, n_bank = 0;
    if (tokens) {
        const bool embedded = count > 1
            ? qw3_metal_inkling_batch_embed(gpu, &embedding, tokens, count, INK_CUR)
            : qw3_metal_inkling_embed(gpu, &embedding, (unsigned)tokens[0], INK_CUR);
        if (!embedded || !inkling_metal_norm(graph, INK_CUR, w->embedding_norm->offset)) return false;
    } else if (embd) {
        /* Raw vision/multimodal embeddings bypass token_embd_norm (matches Inkling2). */
        const bool uploaded = count > 1
            ? qw3_metal_inkling_batch_upload_embd(gpu, embd, count, INK_CUR)
            : qw3_metal_inkling_upload_embd(gpu, embd, INK_CUR);
        if (!uploaded) return false;
    } else {
        return false;
    }

    for (unsigned visit = 0; visit < INKLING_VISITS; visit++) {
        const unsigned physical = inkling_weight_layer(visit);
        const inkling_layer *l = &w->layer[physical];
        const bool sliding = m->inkling.sliding[visit];
        const bool repeat = visit >= INKLING_CORE_END && physical < INKLING_CORE_END;
        if (physical == INKLING_CORE_START) {
            if (!repeat) {
                n_prelude = n_blocks;
                if (!inkling_metal_vector_op(graph, INK_PRELUDE, INK_CUR, 1, false)) return false;
            } else {
                n_bank = n_blocks - n_prelude - 1;
                if (!inkling_metal_vector_op(graph, INK_BANKS, INK_BLOCKS + n_prelude + 1, n_bank, false) ||
                    !inkling_metal_vector_op(graph, INK_BANK_LAST, INK_CUR, 1, false) ||
                    !inkling_metal_vector_op(graph, INK_CUR, INK_PRELUDE, 1, false)) return false;
                n_blocks = n_prelude;
            }
        }
        if (repeat) {
            const uint64_t row = 2 * (physical - INKLING_CORE_START) * row_bytes;
            const uint64_t attn = w->loop_res->offset + row, kv = w->loop_kv_res->offset + row;
            if (!inkling_metal_mix4(graph, INK_MESSAGE, INK_BANK_LAST, INK_BANKS,
                                    n_bank, attn, kv)) return false;
        }
        const qw3_tensor *norm[] = {l->q_norm, l->k_norm, l->v_norm};
        const bool shared_qkv = sliding && !repeat;
        if (shared_qkv) {
            if (l->attn_res) {
                if (!inkling_metal_mix_norm(graph, INK_INPUT, INK_CUR, INK_BLOCKS, n_blocks,
                                            l->attn_res->offset, norm[0]->offset))
                    return false;
            } else if (!inkling_metal_vector_op(graph, INK_INPUT, INK_CUR, 1, false) ||
                       !inkling_metal_norm(graph, INK_INPUT, norm[0]->offset)) {
                return false;
            }
        } else if (sliding && repeat) {
            if (l->attn_res) {
                if (!inkling_metal_mix(graph, INK_INPUT, INK_CUR, INK_BLOCKS, n_blocks, l->attn_res->offset))
                    return false;
            } else if (!inkling_metal_vector_op(graph, INK_INPUT, INK_CUR, 1, false)) {
                return false;
            }
            if (!inkling_apply_metal_control_from(graph, l, 1, INK_INPUT, INK_INPUT + 1) ||
                !inkling_apply_metal_control_from(graph, l, 2, INK_INPUT, INK_INPUT + 2) ||
                !inkling_apply_metal_control_from(graph, l, 0, INK_INPUT, INK_INPUT)) return false;
            for (unsigned j = 0; j < 3; j++) {
                if (!inkling_metal_norm(graph, INK_INPUT + j, norm[j]->offset)) return false;
            }
        } else {
            for (unsigned j = 0; j < 3; j++) {
                const unsigned dst = INK_INPUT + j;
                if (!repeat && l->attn_res) {
                    const uint64_t query = l->attn_res->offset + j * row_bytes;
                    if (!inkling_metal_mix_norm(graph, dst, INK_CUR, INK_BLOCKS, n_blocks,
                                                query, norm[j]->offset)) return false;
                } else {
                    if (l->attn_res) {
                        const uint64_t query = l->attn_res->offset + j * row_bytes;
                        if (!inkling_metal_mix(graph, dst, INK_CUR, INK_BLOCKS, n_blocks, query)) return false;
                    } else if (!inkling_metal_vector_op(graph, dst, INK_CUR, 1, false)) {
                        return false;
                    }
                    if (repeat && !inkling_apply_metal_control(graph, l, j, dst)) return false;
                    if (!inkling_metal_norm(graph, dst, norm[j]->offset)) return false;
                }
            }
        }
        const bool save_block = physical % m->inkling.attn_res_block == 0;
        if (save_block) {
            if (!inkling_metal_vector_op(graph, INK_BLOCKS + n_blocks, INK_CUR, 1, false)) return false;
            n_blocks++;
            if (!inkling_metal_attention(graph, m, l, visit, pos, sliding, shared_qkv, INK_CUR, false))
                return false;
        } else {
            if (!inkling_metal_attention(graph, m, l, visit, pos, sliding, shared_qkv, INK_CUR, true))
                return false;
        }
        bool ffn_added = false;
        if (!repeat) {
            if (!inkling_metal_mix_norm(graph, INK_INPUT, INK_CUR, INK_BLOCKS, n_blocks,
                                        l->ffn_res->offset, l->ffn_norm->offset))
                return false;
        } else {
            if (!inkling_metal_mix(graph, INK_INPUT, INK_CUR, INK_BLOCKS, n_blocks, l->ffn_res->offset) ||
                !inkling_apply_metal_control(graph, l, 3, INK_INPUT) ||
                !inkling_metal_norm(graph, INK_INPUT, l->ffn_norm->offset))
                return false;
        }
        if (!inkling_metal_ffn(graph, m, l, visit, pos, &ffn_added)) return false;
        if (!ffn_added && !inkling_metal_vector_op(graph, INK_CUR, INK_OUT, 1, true)) return false;
    }

    if (count > 1 && !compute_logits) {
        return qw3_metal_inkling_batch_finish(gpu, NULL, 0, count, 0.0f, NULL);
    }
    qw3_metal_projection output = inkling_metal_projection(w->output, 0);
    output.n_out = m->inkling.vocab_unpadded;
    if (!inkling_metal_mix_norm(graph, INK_OUT, INK_CUR, INK_BLOCKS, n_blocks,
                                w->output_res->offset, w->output_norm->offset)) return false;
    return count > 1
        ? qw3_metal_inkling_batch_finish(gpu, &output, INK_OUT, count, m->inkling.logit_cap, s->logits)
        : qw3_metal_inkling_logits(gpu, &output, INK_OUT, m->inkling.logit_cap, s->logits);
}

static int inkling_prefill_metal(qw3_session *s, const int *tokens, int count,
                                 bool compute_logits, char *err, size_t errlen) {
    const int pos = s->tokens.len;
    const unsigned vocab = s->engine->model.inkling.vocab_unpadded;
    if (count > s->ctx_size - pos) {
        if (err && errlen) snprintf(err, errlen, "Inkling batch exceeds the session context");
        return -1;
    }
    for (int i = 0; i < count; i++) {
        if ((unsigned)tokens[i] >= vocab) {
            if (err && errlen) snprintf(err, errlen, "Inkling token is outside the vocabulary");
            return -1;
        }
    }
    for (unsigned i = 0; i < INKLING_LAYERS; i++) {
        const inkling_layer *layer = &s->engine->inkling->layer[i];
        if (!inkling_mapped_expert_type(layer->gate_up->type) ||
            !inkling_mapped_expert_type(layer->down->type)) return 1;
    }
    if (!qw3_metal_inkling_batch_begin(s->inkling->metal, (unsigned)count)) return 1;
    if (!inkling_eval_metal(s, tokens, NULL, (unsigned)count, (unsigned)pos, compute_logits)) {
        qw3_metal_inkling_discard(s->inkling->metal);
        qw3_session_invalidate(s);
        if (err && errlen) snprintf(err, errlen, "Inkling Metal batch failed at token %d", pos);
        return -1;
    }
    for (int i = 0; i < count; i++) token_vec_push(&s->tokens, tokens[i]);
    s->kv.pos += count;
    s->valid = true;
    s->cpu_logits_valid = compute_logits;
    return 0;
}

static int inkling_prefill_embd_metal(qw3_session *s, const float *embd,
                                      const int *sentinels, int count,
                                      bool compute_logits, char *err, size_t errlen) {
    const int pos = s->tokens.len;
    if (count > s->ctx_size - pos) {
        if (err && errlen) snprintf(err, errlen, "Inkling embedding batch exceeds the session context");
        return -1;
    }
    if (count > 1) {
        for (unsigned i = 0; i < INKLING_LAYERS; i++) {
            const inkling_layer *layer = &s->engine->inkling->layer[i];
            if (!inkling_mapped_expert_type(layer->gate_up->type) ||
                !inkling_mapped_expert_type(layer->down->type)) return 1;
        }
        if (!qw3_metal_inkling_batch_begin(s->inkling->metal, (unsigned)count)) return 1;
    }
    if (!inkling_eval_metal(s, NULL, embd, (unsigned)count, (unsigned)pos, compute_logits)) {
        qw3_metal_inkling_discard(s->inkling->metal);
        qw3_session_invalidate(s);
        if (err && errlen) snprintf(err, errlen, "Inkling Metal embedding batch failed at pos %d", pos);
        return -1;
    }
    for (int i = 0; i < count; i++) token_vec_push(&s->tokens, sentinels ? sentinels[i] : -1);
    s->kv.pos += count;
    s->valid = true;
    s->cpu_logits_valid = compute_logits || (count == 1);
    return 0;
}
#endif

static int inkling_eval_internal(qw3_session *s, int token, const float *raw_embd,
                                 char *err, size_t errlen) {
    const qw3_model *m = &s->engine->model;
    const qw3_inkling_weights *w = s->engine->inkling;
    const unsigned pos = (unsigned)s->tokens.len;
    if (pos >= (unsigned)s->ctx_size ||
        (!raw_embd && (token < 0 || token >= m->inkling.vocab_unpadded))) {
        if (err && errlen) snprintf(err, errlen, "Inkling token or position is outside the session");
        return -1;
    }
    if (!s->valid) {
        if (pos != 0) {
            if (err && errlen) snprintf(err, errlen, "Inkling requires replay after rewind");
            return -1;
        }
        inkling_state_reset(s->inkling);
    }
#ifndef QW3_NO_METAL
    if (s->inkling->metal) {
        if (!inkling_eval_metal(s, raw_embd ? NULL : &token, raw_embd, 1, pos, true)) {
            qw3_metal_inkling_discard(s->inkling->metal);
            goto failed;
        }
        goto complete;
    }
#endif
    float cur[INK_D], out[INK_D], input[3][INK_D], messages[4][INK_D];
    float blocks[INK_RES_BLOCKS][INK_D], prelude[INK_D], bank[INK_RES_BLOCKS][INK_D], bank_last[INK_D];
    unsigned n_blocks = 0, n_prelude = 0, n_bank = 0;
    if (raw_embd) {
        memcpy(cur, raw_embd, sizeof(cur));
    } else {
        if (!tensor_read_dense_row(m, w->embedding, token, cur)) goto failed;
        cpu_rmsnorm(cur, cur, m, w->embedding_norm, INK_D);
    }

    for (unsigned visit = 0; visit < INKLING_VISITS; visit++) {
        const unsigned physical = inkling_weight_layer(visit);
        const inkling_layer *l = &w->layer[physical];
        inkling_visit_state *state = &s->inkling->visit[visit];
        const bool sliding = m->inkling.sliding[visit];
        const bool repeat = visit >= INKLING_CORE_END && physical < INKLING_CORE_END;
        if (physical == INKLING_CORE_START) {
            if (!repeat) {
                n_prelude = n_blocks;
                memcpy(prelude, cur, sizeof(prelude));
            } else {
                n_bank = n_blocks - n_prelude - 1;
                memcpy(bank, blocks + n_prelude + 1, n_bank * sizeof(bank[0]));
                memcpy(bank_last, cur, sizeof(bank_last));
                n_blocks = n_prelude;
                memcpy(cur, prelude, sizeof(cur));
            }
        }
        if (repeat) {
            const unsigned row = 2 * (physical - INKLING_CORE_START);
            const float *attn = inkling_f32(m, w->loop_res) + row * INK_D;
            const float *kv = inkling_f32(m, w->loop_kv_res) + row * INK_D;
            inkling_mix(messages[0], bank_last, bank, n_bank, attn);
            inkling_mix(messages[1], bank_last, bank, n_bank, kv);
            inkling_mix(messages[2], bank_last, bank, n_bank, kv + INK_D);
            inkling_mix(messages[3], bank_last, bank, n_bank, attn + INK_D);
        }
        const qw3_tensor *norm[3] = {l->q_norm, l->k_norm, l->v_norm};
        for (unsigned j = 0; j < 3; j++) {
            if (l->attn_res) {
                const float *query = inkling_f32(m, l->attn_res) + (sliding ? 0 : j * INK_D);
                inkling_mix(input[j], cur, blocks, n_blocks, query);
            } else {
                memcpy(input[j], cur, sizeof(cur));
            }
        }
        if (repeat && !inkling_control(s, l, 0, 3, input, messages)) goto failed;
        for (unsigned j = 0; j < 3; j++) {
            cpu_rmsnorm(input[j], input[j], m, norm[j], INK_D);
        }
        if (!inkling_attention(s, l, visit, pos, sliding, input, out)) goto failed;
        if (physical % m->inkling.attn_res_block == 0) {
            memcpy(blocks[n_blocks++], cur, sizeof(cur));
            memcpy(cur, out, sizeof(cur));
        } else {
            for (unsigned j = 0; j < INK_D; j++) cur[j] += out[j];
        }
        inkling_mix(input[0], cur, blocks, n_blocks, inkling_f32(m, l->ffn_res));
        if (repeat && !inkling_control(s, l, 3, 1, input, messages + 3)) goto failed;
        cpu_rmsnorm(input[0], input[0], m, l->ffn_norm, INK_D);
        if (!inkling_ffn(s, l, input[0], out)) goto failed;
        inkling_conv(out, INK_D, inkling_f32(m, l->ffn_conv), NULL, state, 2 * INK_KV + INK_D);
        for (unsigned j = 0; j < INK_D; j++) cur[j] += out[j];
    }

    inkling_mix(out, cur, blocks, n_blocks, inkling_f32(m, w->output_res));
    cpu_rmsnorm(out, out, m, w->output_norm, INK_D);
    qw3_tensor output = *w->output;
    output.dim[1] = m->inkling.vocab_unpadded;
    if (!inkling_matvec(s, &output, out, s->logits)) goto failed;
    const float cap = m->inkling.logit_cap;
    for (unsigned i = 0; i < output.dim[1]; i++) s->logits[i] = cap * tanhf(s->logits[i] / cap);
#ifndef QW3_NO_METAL
complete:
#endif
    token_vec_push(&s->tokens, token);
    s->kv.pos++;
    s->valid = s->cpu_logits_valid = true;
    return 0;

failed:
    qw3_session_invalidate(s);
    if (err && errlen) snprintf(err, errlen, "Inkling projection failed");
    return -1;
}

static int inkling_eval(qw3_session *s, int token, char *err, size_t errlen) {
    return inkling_eval_internal(s, token, NULL, err, errlen);
}

static int inkling_eval_embd_one(qw3_session *s, const float *embd, int sentinel,
                                 char *err, size_t errlen) {
    return inkling_eval_internal(s, sentinel, embd, err, errlen);
}
