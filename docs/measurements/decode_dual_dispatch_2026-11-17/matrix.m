#include "adapter.inc"

static int compare_pair(qw3_metal_session *s, const qw3_tensor *a, const qw3_tensor *b) {
    uint32_t n_in = (uint32_t)a->dim[0];
    uint32_t na = (uint32_t)a->dim[1], nb = (uint32_t)b->dim[1];
    uint32_t rows = na + nb;
    float *ref = calloc(rows, sizeof(float)), *actual = calloc(rows, sizeof(float));
    if (!ref || !actual) abort();
    int ok = qw3_metal_session_matvec_quant_x1_to_scratch(s, a->offset, a->type, n_in, na, 0, ref) &&
             qw3_metal_session_matvec_quant_x1_to_scratch(s, b->offset, b->type, n_in, nb, na, ref + na) &&
             dual_matvec(s, a->offset, b->offset, a->type, b->type, n_in, na, nb) &&
             qw3_metal_session_read_scratch(s, actual, 0, rows);
    int exact = ok;
    for (uint32_t part = 0; ok && part < 2; part++) {
        unsigned mismatches = 0;
        uint32_t start = part ? na : 0;
        uint32_t end = part ? rows : na;
        for (uint32_t i = start; i < end; i++) {
            if (!isfinite(actual[i]) || memcmp(&ref[i], &actual[i], sizeof(float))) {
                if (!mismatches)
                    printf("first mismatch part=%u row=%u baseline=%.9g candidate=%.9g\n",
                           part, i-start, ref[i], actual[i]);
                mismatches++;
            }
        }
        printf("matrix part=%u type=%u rows=%u differences=%u\n",
               part, part ? b->type : a->type, end-start, mismatches);
        exact &= mismatches == 0;
    }
    free(ref);
    free(actual);
    return exact;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        qw3_engine *e = NULL;
        qw3_session *s = NULL;
        qw3_engine_options options = {.model_path=argv[1], .backend=QW3_BACKEND_METAL, .n_threads=4};
        int rc = 1;
        if (qw3_engine_open(&e, &options) || qw3_session_create(&s, e, 256)) goto done;
        if (!qw3_metal_session_embed_q3_k(s->metal, e->weights.token_embd->offset, 66, 5120, NULL) ||
            !qw3_metal_session_rmsnorm_weight_f32(s->metal, e->weights.layer[0].attn_norm->offset,
                                                5120, QW3_RMS_EPS, NULL)) goto done;
        dual_mode = 3;
        unsigned seen[2][2] = {{0}}, pass = 0, count = 0;
        for (int l = 0; l < qw3_engine_layer_count(e); l++) {
            const qw3_layer_weights *w = &e->weights.layer[l];
            const qw3_tensor *a = w->linear_qkv_proj, *b = w->linear_gate_proj;
            if (!a || !b || !dual_enabled(a->type, b->type) || seen[a->type==23][b->type==23]++) continue;
            count++;
            printf("matrix layer=%d types=%u/%u\n", l, a->type, b->type);
            pass += compare_pair(s->metal, a, b);
        }
        printf("matrix cases=%u pass=%u\n", count, pass);
        rc = count == 4 && pass == count ? 0 : 1;
done:
        if (s) qw3_session_free(s);
        if (e) qw3_engine_close(e);
        return rc;
    }
}
