#include "qw3.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then builds an independent decode session for that same frontier and
 * performs a fixed greedy decode run without allowing EOS.  Decode prefill
 * setup is intentionally outside both timing windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    qw3_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int n_prompt;
    int gen_tokens;
    int depth;
    int repetitions;
    double step_mul;
    const char *dump_frontier_logits_dir;
    uint32_t seed;
    bool llama_style;
    bool no_warmup;
    bool warm_weights;
    bool quality;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: qw3-bench --prompt-file FILE [options]\n"
        "\n"
        "Benchmarks instantaneous prefill and generation throughput at context\n"
        "frontiers such as 2048, 4096, 6144, ... . Generation is always greedy,\n"
        "runs for exactly --gen-tokens tokens, and skips EOS so every row is\n"
        "comparable.\n"
        "\n"
        "Input:\n"
        "  --prompt-file FILE\n"
        "      Raw benchmark text. The fixed token sequence is sliced at each frontier.\n"
        "  --chat-prompt-file FILE\n"
        "      Render FILE as one no-thinking chat user message, then slice that sequence.\n"
        "  -sys, --system TEXT\n"
        "      System prompt used only with --chat-prompt-file.\n"
        "  --llama-style\n"
        "      Run a synthetic llama-bench-like benchmark instead of reading a prompt.\n"
        "\n"
        "Model and backend:\n"
        "  -m, --model FILE       GGUF model path. Default: qw3.gguf\n"
        "  --metal | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS, CPU elsewhere.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Ignored by qw3-bench.\n"
        "  --warm-weights         Touch mapped tensor pages before benchmarking.\n"
        "\n"
        "Sweep:\n"
        "  --ctx-start N          First measured frontier. Default: 2048\n"
        "  --ctx-max N            Last measured frontier. Default: 32768\n"
        "  --ctx-alloc N          Allocated context. Default: ctx-max + gen-tokens + 1\n"
        "  --step-mul F           Multiplicative step. Default: 1\n"
        "  --step-incr N          Linear step when --step-mul is 1. Default: 2048\n"
        "  --gen-tokens N         Greedy decode tokens per frontier. Default: 128\n"
        "\n"
        "Llama-style synthetic mode:\n"
        "  -p, --n-prompt N       Prompt-processing tokens. Default: 512\n"
        "  -n, --n-gen N          Token-generation tokens. Default: 128\n"
        "                         When both are non-zero, they are measured separately.\n"
        "  -d, --depth N          Prefilled context before the timed run. Default: 0\n"
        "  -r, --repetitions N    Timed repetitions. Default: 5\n"
        "  --no-warmup            Skip untimed warmup run.\n"
        "  --seed N               Synthetic token seed. Default: 1\n"
        "\n"
        "Output:\n"
        "  --csv FILE             Write CSV there instead of stdout.\n"
        "  --dump-frontier-logits-dir DIR\n"
        "      Write one top-logit JSON file per measured frontier. DIR must exist.\n"
        "  -h, --help             Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "qw3-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "qw3-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "qw3-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "qw3-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static qw3_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return QW3_BACKEND_METAL;
    if (!strcmp(s, "cpu")) return QW3_BACKEND_CPU;
    fprintf(stderr, "qw3-bench: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "qw3-bench: valid backends are: metal, cpu\n");
    exit(2);
}

static qw3_backend default_backend(void) {
#ifdef __APPLE__
    return QW3_BACKEND_METAL;
#else
    return QW3_BACKEND_CPU;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "qw3-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "qw3-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "qw3-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "qw3-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "qw3-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "qw3-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "qw3.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .n_prompt = 512,
        .gen_tokens = 128,
        .repetitions = 5,
        .step_mul = 1.0,
        .seed = 1,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--llama-style")) {
            c.llama_style = true;
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--n-prompt")) {
            c.n_prompt = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") ||
                   !strcmp(arg, "-n") || !strcmp(arg, "--n-gen")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-d") || !strcmp(arg, "--depth")) {
            c.depth = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-r") || !strcmp(arg, "--repetitions")) {
            c.repetitions = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--no-warmup")) {
            c.no_warmup = true;
        } else if (!strcmp(arg, "--seed")) {
            c.seed = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = QW3_BACKEND_METAL;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = QW3_BACKEND_CPU;
        } else if (!strcmp(arg, "--cuda")) {
            fprintf(stderr, "qw3-bench: CUDA backend is not supported\n");
            exit(2);
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else {
            fprintf(stderr, "qw3-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (c.llama_style) {
        if (c.prompt_path || c.chat_prompt_path) {
            fprintf(stderr, "qw3-bench: --llama-style does not use --prompt-file or --chat-prompt-file\n");
            exit(2);
        }
        if (c.n_prompt == 0 && c.gen_tokens == 0) {
            fprintf(stderr, "qw3-bench: --llama-style needs non-zero --n-prompt or --n-gen\n");
            exit(2);
        }
        if (c.n_prompt > INT_MAX - c.gen_tokens ||
            c.depth > INT_MAX - c.n_prompt - c.gen_tokens - 1) {
            fprintf(stderr, "qw3-bench: requested synthetic context is too large\n");
            exit(2);
        }
        int needed = c.depth + c.n_prompt + c.gen_tokens;
        if (c.ctx_alloc == 0) c.ctx_alloc = needed + 1;
        if (c.ctx_alloc <= needed) {
            fprintf(stderr, "qw3-bench: --ctx-alloc must be greater than depth + n-prompt + n-gen\n");
            exit(2);
        }
    } else {
        if (!!c.prompt_path == !!c.chat_prompt_path) {
            fprintf(stderr, "qw3-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
            exit(2);
        }
        if (c.ctx_start > c.ctx_max) {
            fprintf(stderr, "qw3-bench: --ctx-start must be <= --ctx-max\n");
            exit(2);
        }
        if (c.step_mul < 1.0) {
            fprintf(stderr, "qw3-bench: --step-mul must be >= 1\n");
            exit(2);
        }
        if (c.step_mul == 1.0 && c.step_incr <= 0) {
            fprintf(stderr, "qw3-bench: --step-incr must be positive when --step-mul is 1\n");
            exit(2);
        }
        if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
            fprintf(stderr, "qw3-bench: requested context is too large\n");
            exit(2);
        }
        if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
        if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
            fprintf(stderr, "qw3-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
            exit(2);
        }
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        qw3_session         *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int top_k = 64;
    qw3_token_score *scores = malloc((size_t)top_k * sizeof(scores[0]));
    if (!scores) {
        fprintf(stderr, "qw3-bench: out of memory copying frontier logits\n");
        return 1;
    }
    const int n = qw3_session_top_logprobs(session, scores, top_k);

    char path[PATH_MAX];
    const int len = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.top_logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (len <= 0 || (size_t)len >= sizeof(path)) {
        fprintf(stderr, "qw3-bench: frontier logits path is too long\n");
        free(scores);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "qw3-bench: failed to open %s: %s\n", path, strerror(errno));
        free(scores);
        return 1;
    }

    const int argmax = qw3_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"qw3-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"prompt_tokens\":%d,\n  \"frontier_tokens\":%d,\n"
            "  \"prefill_tokens\":%d,\n  \"ctx\":%d,\n"
            "  \"argmax_id\":%d,\n  \"top_logits\":[\n",
            qw3_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            argmax);

    for (int i = 0; i < n; i++) {
        fprintf(fp,
                "    {\"id\":%d,\"logit\":%.9g}%s\n",
                scores[i].id,
                scores[i].logit,
                i + 1 < n ? "," : "");
    }
    fputs("  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "qw3-bench: failed to close %s\n", path);
        free(scores);
        return 1;
    }
    free(scores);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(qw3_backend backend, int ctx_size) {
    qw3_context_memory m = qw3_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "qw3-bench: context buffers %.2f MiB (ctx=%d, backend=%s, gqa_kv=%.2f MiB, deltanet=%.2f MiB, scratch=%.2f MiB)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            qw3_backend_name(backend),
            (double)m.gqa_kv_bytes / (1024.0 * 1024.0),
            (double)m.deltanet_state_bytes / (1024.0 * 1024.0),
            (double)m.scratch_bytes / (1024.0 * 1024.0));
}

static int qw3_session_argmax_excluding(qw3_session *session, int exclude) {
    if (!session) return -1;
    const int top_k = 64;
    qw3_token_score *scores = malloc((size_t)top_k * sizeof(scores[0]));
    if (!scores) return -1;
    int n = qw3_session_top_logprobs(session, scores, top_k);
    int token = -1;
    for (int i = 0; i < n; i++) {
        if (scores[i].id != exclude) {
            token = scores[i].id;
            break;
        }
    }
    free(scores);
    return token;
}

static uint32_t bench_rng_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static int bench_random_token(uint32_t *state, int n_vocab, int eos) {
    if (n_vocab <= 1) return -1;
    int token = (int)(bench_rng_next(state) % (uint32_t)n_vocab);
    if (token == eos) token = (token + 1) % n_vocab;
    return token;
}

static int fill_synthetic_tokens(qw3_tokens *tokens, int n_tokens,
                                 int n_vocab, int eos, uint32_t seed) {
    if (!tokens || n_tokens < 0 || n_vocab <= 1) return -1;
    uint32_t rng = seed ? seed : 1u;
    for (int i = 0; i < n_tokens; i++) {
        int token = bench_random_token(&rng, n_vocab, eos);
        if (token < 0) return -1;
        qw3_tokens_push(tokens, token);
    }
    return 0;
}

static double avg_double(const double *v, int n) {
    if (!v || n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i];
    return sum / (double)n;
}

static double stdev_double(const double *v, int n) {
    if (!v || n <= 1) return 0.0;
    double mean = avg_double(v, n);
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        double d = v[i] - mean;
        ss += d * d;
    }
    return sqrt(ss / (double)(n - 1));
}

static int run_llama_style_case(qw3_engine *engine, const bench_config *cfg,
                                int n_prompt, int n_gen, bool print_header) {
    if (!engine || !cfg) return 1;
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

        const double t0 = bench_now_sec();
        if (n_prompt > 0 &&
            qw3_session_sync(session, &prompt_prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "qw3-bench: prompt run failed: %s\n", err);
            rc = 1;
            break;
        }
        for (int i = 0; i < n_gen; i++) {
            int token = synthetic.v[cfg->depth + n_prompt + i];
            if (qw3_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "qw3-bench: generation run failed: %s\n", err);
                rc = 1;
                break;
            }
        }
        const double t1 = bench_now_sec();
        if (rc) break;

        samples_sec[rep] = t1 - t0;
        samples_tps[rep] = samples_sec[rep] > 0.0 ?
            (double)timed_tokens / samples_sec[rep] : 0.0;
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

static int run_llama_style_bench(qw3_engine *engine, const bench_config *cfg) {
    bool printed = false;
    if (cfg->n_prompt > 0) {
        int rc = run_llama_style_case(engine, cfg, cfg->n_prompt, 0, !printed);
        printed = true;
        if (rc != 0) return rc;
    }
    if (cfg->gen_tokens > 0) {
        int rc = run_llama_style_case(engine, cfg, 0, cfg->gen_tokens, !printed);
        printed = true;
        if (rc != 0) return rc;
    }
    return 0;
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);
    log_context_memory(cfg.backend, cfg.ctx_alloc);

    qw3_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .warm_weights = cfg.warm_weights,
    };
    qw3_engine *engine = NULL;
    if (qw3_engine_open(&engine, &opt) != 0) return 1;

    if (cfg.llama_style) {
        int rc = run_llama_style_bench(engine, &cfg);
        qw3_engine_close(engine);
        return rc;
    }

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    qw3_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        qw3_encode_chat_prompt(engine, cfg.system, text, QW3_THINK_NONE, &prompt);
    } else {
        qw3_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "qw3-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        qw3_tokens_free(&prompt);
        qw3_engine_close(engine);
        return 1;
    }

    qw3_session *prefill_session = NULL;
    if (qw3_session_create(&prefill_session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "qw3-bench: failed to create prefill session\n");
        qw3_tokens_free(&prompt);
        qw3_engine_close(engine);
        return 1;
    }
    qw3_session *decode_session = NULL;

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "qw3-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            qw3_session_free(decode_session);
            qw3_session_free(prefill_session);
            qw3_tokens_free(&prompt);
            qw3_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,session_bytes\n");
    fflush(out);

    const int eos = qw3_token_eos(engine);
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        qw3_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = prompt.cap,
        };

        const double prefill_t0 = bench_now_sec();
        if (qw3_session_sync(prefill_session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "qw3-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;
        const uint64_t session_bytes = qw3_session_payload_bytes(prefill_session);

        if (write_frontier_logits_json(&cfg, prefill_session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        qw3_session *gen_session = prefill_session;
        if (frontier < cfg.ctx_max) {
            if (!decode_session &&
                qw3_session_create(&decode_session, engine, cfg.ctx_alloc) != 0) {
                fprintf(stderr, "qw3-bench: failed to create decode session\n");
                rc = 1;
                break;
            }
            qw3_session_invalidate(decode_session);
            if (qw3_session_sync(decode_session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "qw3-bench: decode setup at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            gen_session = decode_session;
        }

        const double gen_t0 = bench_now_sec();
        for (int i = 0; i < cfg.gen_tokens; i++) {
            if (qw3_session_pos(gen_session) + 1 >= qw3_session_ctx(gen_session)) {
                fprintf(stderr, "qw3-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = qw3_session_argmax_excluding(gen_session, eos);
            if (token < 0) {
                fprintf(stderr, "qw3-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (qw3_session_eval(gen_session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "qw3-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (rc != 0) break;

        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                cfg.gen_tokens,
                gen_sec > 0.0 ? (double)cfg.gen_tokens / gen_sec : 0.0,
                (unsigned long long)session_bytes);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    qw3_session_free(decode_session);
    qw3_session_free(prefill_session);
    qw3_tokens_free(&prompt);
    qw3_engine_close(engine);
    return rc;
}
