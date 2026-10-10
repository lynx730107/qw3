#include "projection_candidates.inc"
#define main probe_native_main
#include "../qw3_bench.c"
#undef main
static int probe_run_case(qw3_engine *engine, const bench_config *cfg,
                                int n_prompt, int n_gen, bool print_header) {
    const int profile=getenv("QW3_PRIVATE_STAGE_PROFILE")!=NULL;
    if (!engine || !cfg || (profile && !probe_profile_initialize())) return 1;
    const int n_vocab = qw3_vocab_size(engine);
    const int eos = qw3_token_eos(engine);
    const int timed_tokens = n_prompt + n_gen;
    const int total_tokens = cfg->depth + timed_tokens;
    qw3_tokens synthetic = {0};
    char err[256];
    int rc = 0;

    if (fill_synthetic_tokens(&synthetic, total_tokens, n_vocab, eos,
                              cfg->seed) != 0) {
        fprintf(stderr, "qw3-bench: failed to create synthetic token stream\n");
        return 1;
    }

    qw3_session *session = NULL;
    if (qw3_session_create(&session, engine, cfg->ctx_alloc) != 0 || !session) {
        fprintf(stderr, "qw3-bench: failed to create llama-style session\n");
        qw3_tokens_free(&synthetic);
        return 1;
    }

    qw3_tokens depth_prefix = {
        .v = synthetic.v,
        .len = cfg->depth,
        .cap = synthetic.cap,
    };
    qw3_tokens prompt_prefix = {
        .v = synthetic.v,
        .len = cfg->depth + n_prompt,
        .cap = synthetic.cap,
    };

    if (!cfg->no_warmup) {
        qw3_session_invalidate(session);
        /* Recreate the same depth used by the measured generation, then warm
         * one token. Depth preparation remains outside the measured timer. */
        if (cfg->depth > 0 &&
            qw3_session_sync(session, &depth_prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "qw3-bench: warmup depth failed: %s\n", err);
            rc = 1;
        }
        if (!rc && n_prompt > 0 &&
            qw3_session_sync(session, &prompt_prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "qw3-bench: warmup prompt failed: %s\n", err);
            rc = 1;
        }
        if (!rc && n_gen > 0) {
            int token = synthetic.v[cfg->depth + n_prompt];
            if (qw3_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "qw3-bench: warmup generation failed: %s\n", err);
                rc = 1;
            }
        }
    }

    double *samples_sec = NULL;
    double *samples_tps = NULL;
    if (!rc) {
        samples_sec = calloc((size_t)cfg->repetitions, sizeof(samples_sec[0]));
        samples_tps = calloc((size_t)cfg->repetitions, sizeof(samples_tps[0]));
        if (!samples_sec || !samples_tps) {
            fprintf(stderr, "qw3-bench: out of memory for timing samples\n");
            rc = 1;
        }
    }

    uint64_t session_bytes = 0;
    for (int rep = 0; !rc && rep < cfg->repetitions; rep++) {
        qw3_session_invalidate(session);
        if (cfg->depth > 0 &&
            qw3_session_sync(session, &depth_prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "qw3-bench: depth setup failed: %s\n", err);
            rc = 1;
            break;
        }

        if (profile) probe_profile_start();
        const double t0 = bench_now_sec();
        if (n_prompt > 0 &&
            qw3_session_sync(session, &prompt_prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "qw3-bench: prompt run failed: %s\n", err);
            rc = 1;
            break;
        }
        const char *chunk_env = getenv("QW3_BENCH_PRINT_CHUNKS");
        int chunk_tokens = chunk_env ? atoi(chunk_env) : 0;
        double chunk_start = bench_now_sec();
        for (int i = 0; i < n_gen; i++) {
            int token = synthetic.v[cfg->depth + n_prompt + i];
            if (qw3_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "qw3-bench: generation run failed: %s\n", err);
                rc = 1;
                break;
            }
            if (chunk_tokens > 0 &&
                ((i + 1) % chunk_tokens == 0 || i + 1 == n_gen)) {
                double chunk_end = bench_now_sec();
                int count = (i + 1) % chunk_tokens;
                if (count == 0) count = chunk_tokens;
                fprintf(stderr,
                        "qw3-bench: generation tokens %d..%d %.3f tok/s %.3f ms\n",
                        i + 2 - count, i + 1, count / (chunk_end - chunk_start),
                        (chunk_end - chunk_start) * 1000.0);
                chunk_start = chunk_end;
            }
        }
        const double t1 = bench_now_sec();
        if (profile) probe_profile_end(rep,t1-t0);
        if (rc) break;

        samples_sec[rep] = t1 - t0;
        samples_tps[rep] = samples_sec[rep] > 0.0 ?
            (double)timed_tokens / samples_sec[rep] : 0.0;
        if (getenv("QW3_BENCH_PRINT_SAMPLES")) {
            fprintf(stderr, "qw3-bench: sample %d/%d %.3f tok/s %.3f ms\n",
                    rep + 1, cfg->repetitions, samples_tps[rep],
                    samples_sec[rep] * 1000.0);
        }
        session_bytes = qw3_session_payload_bytes(session);
    }

    if (!rc) {
        const char *kind = n_prompt > 0 && n_gen > 0 ? "pp+tg" :
            (n_prompt > 0 ? "pp" : "tg");
        if (print_header) {
            printf("test,n_prompt,n_gen,n_depth,reps,avg_tps,stdev_tps,avg_ms,stdev_ms,session_bytes\n");
        }
        printf("%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%llu\n",
               kind,
               n_prompt,
               n_gen,
               cfg->depth,
               cfg->repetitions,
               avg_double(samples_tps, cfg->repetitions),
               stdev_double(samples_tps, cfg->repetitions),
               avg_double(samples_sec, cfg->repetitions) * 1000.0,
               stdev_double(samples_sec, cfg->repetitions) * 1000.0,
               (unsigned long long)session_bytes);
    }

    free(samples_tps);
    free(samples_sec);
    qw3_session_free(session);
    qw3_tokens_free(&synthetic);
    return rc;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        bench_config c=parse_options(argc,argv);
        if (!c.llama_style || (c.n_prompt && c.gen_tokens) || c.dump_frontier_logits_dir) return 2;
        char ngl[16];snprintf(ngl,sizeof(ngl),"%d",c.ngl);if(c.ngl_set) setenv("QW3_METAL_NGL",ngl,1);
        qw3_engine *e=NULL;
        qw3_engine_options o={.model_path=c.model_path,.backend=c.backend,.n_threads=c.threads,.warm_weights=c.warm_weights};
        if (qw3_engine_open(&e,&o)) return 1;
        int rc=probe_run_case(e,&c,c.n_prompt,c.gen_tokens,true);
        qw3_engine_close(e);
        fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 q3=%llu iq=%llu iq3=%llu iq4=%llu\n",(unsigned long long)probe_q3_calls,(unsigned long long)probe_iq_calls,
            (unsigned long long)probe_iq3_calls,(unsigned long long)probe_iq4_calls);
        return rc;
    }
}
