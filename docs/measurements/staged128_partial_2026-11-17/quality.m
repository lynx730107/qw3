#include "qw3_metal.m"
#define qw3_metal_env_n_gpu_layers partial_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled partial_core_llama_split_enabled
#define qw3_metal_llama_split_start partial_core_llama_split_start
#include "qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
#include "gqa_heads_numeric_oracles.inc"

static int checkpoint_exact(qw3_session *session, int vocab, char *err, size_t capacity) {
    FILE *reference = tmpfile();
    FILE *payload = tmpfile();
    uint64_t bytes = qw3_session_payload_bytes(session);
    int ok = reference && payload && grid_oracle_logits(reference, 1, session, vocab) &&
        grid_oracle_state(reference, 1, session) &&
        !qw3_session_save_payload(session, payload, err, capacity) &&
        !fseek(payload, 0, SEEK_SET) && !fseek(reference, 0, SEEK_SET);
    if (ok) {
        qw3_session_invalidate(session);
        grid_require_exact_state = 1;
        ok = !qw3_session_load_payload(session, payload, bytes, err, capacity) &&
            grid_oracle_logits(reference, 0, session, vocab) &&
            grid_oracle_state(reference, 0, session) && fgetc(reference) == EOF &&
            !ferror(reference);
        grid_require_exact_state = 0;
    }
    if (payload) fclose(payload);
    if (reference) fclose(reference);
    return ok;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 4) return 2;
        unsigned mode = (unsigned)atoi(argv[2]);
        int steps = atoi(argv[3]);
        if (!mode || mode > 3 || steps < 0 || steps > 32) return 2;
        qw3_engine *engine = NULL;
        qw3_session *session = NULL;
        qw3_tokens tokens = {0};
        qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL,
                                      .n_threads = 4};
        char err[512] = {0};
        int rc = 1, trial = 0, phase = 0, step = -1;
        FILE *oracle = NULL;
        if (qw3_engine_open(&engine, &options) || qw3_session_create(&session, engine, 256) ||
            session->metal_n_gpu_layers != 64) goto done;
        int vocab = qw3_vocab_size(engine);
        uint32_t seed = 1;
        for (int i = 0; i < 224; i++) {
            seed = seed * 1664525u + 1013904223u;
            int token = (int)(seed % (uint32_t)vocab);
            if (token == qw3_token_eos(engine)) token = (token + 1) % vocab;
            qw3_tokens_push(&tokens, token);
        }
        const int lengths[] = {96, 97, 127, 128, 129, 159, 224};
        int cases = argc == 4 ? 7 : 1;
        for (trial = 0; trial < cases; trial++) {
            int length = argc == 4 ? lengths[trial] : atoi(argv[4]);
            if (length < 1 || length > tokens.len) goto done;
            qw3_tokens prefix = {.v = tokens.v, .len = length, .cap = tokens.cap};
            oracle = tmpfile();
            if (!oracle) goto done;
            for (phase = 0; phase < 2; phase++) {
                partial_mode = phase ? mode : 0;
                if (phase && fseek(oracle, 0, SEEK_SET)) goto done;
                qw3_session_invalidate(session);
                step = -1;
                if (qw3_session_sync(session, &prefix, err, sizeof(err)) ||
                    !grid_oracle_logits(oracle, !phase, session, vocab) ||
                    !grid_oracle_state(oracle, !phase, session) ||
                    !checkpoint_exact(session, vocab, err, sizeof(err))) goto done;
                for (step = 0; step < steps; step++) {
                    int token = phase ? 0 : qw3_session_argmax(session);
                    if (phase ? fread(&token, sizeof(token), 1, oracle) != 1
                              : fwrite(&token, sizeof(token), 1, oracle) != 1) goto done;
                    if (token < 0 || qw3_session_eval_gpu_logits(session, token, err, sizeof(err)) ||
                        !grid_oracle_logits(oracle, !phase, session, vocab)) goto done;
                }
                if (!grid_oracle_state(oracle, !phase, session) ||
                    (phase && (fgetc(oracle) != EOF || ferror(oracle)))) goto done;
            }
            fclose(oracle);
            oracle = NULL;
            printf("PASS length=%d mode=%u continuation=%d checkpoint=exact\n", length, mode, steps);
        }
        rc = 0;
done:
        if (rc) fprintf(stderr, "partial staged quality failed trial=%d phase=%d step=%d %s\n",
                        trial, phase, step, err);
        grid_numeric_report();
        if (oracle) fclose(oracle);
        if (session) qw3_session_free(session);
        if (engine) qw3_engine_close(engine);
        qw3_tokens_free(&tokens);
        return rc;
    }
}
