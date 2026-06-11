/* =========================================================================
 * qw3_cli.c — Minimal CLI for Qwen3.6-35B-A3B inference.
 * =========================================================================
 *
 * Usage:
 *   ./qw3 -m model.gguf -p "prompt"
 *   ./qw3 -m model.gguf                  (interactive mode)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include "qw3.h"

#ifndef QW3_CLI_ENABLE_INTERNAL_TESTS
#define QW3_CLI_ENABLE_INTERNAL_TESTS 1
#endif

#define QW3_CLI_N_LAYER 40

typedef struct
{
    qw3_engine *engine;
    qw3_tokens *capture;
} emit_ctx;

typedef struct
{
    float temperature;
    int sample_top_k;
    float top_p;
    float min_p;
    uint64_t rng;
} sample_opts;

static int utf8_read_cp(const char *s, size_t len, size_t *pos, uint32_t *cp)
{
    unsigned char c = (unsigned char)s[*pos];
    if (c < 0x80)
    {
        *cp = c;
        (*pos)++;
        return 1;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len)
    {
        *cp = ((uint32_t)(c & 0x1f) << 6) |
              ((uint32_t)((unsigned char)s[*pos + 1] & 0x3f));
        *pos += 2;
        return 1;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len)
    {
        *cp = ((uint32_t)(c & 0x0f) << 12) |
              ((uint32_t)((unsigned char)s[*pos + 1] & 0x3f) << 6) |
              ((uint32_t)((unsigned char)s[*pos + 2] & 0x3f));
        *pos += 3;
        return 1;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len)
    {
        *cp = ((uint32_t)(c & 0x07) << 18) |
              ((uint32_t)((unsigned char)s[*pos + 1] & 0x3f) << 12) |
              ((uint32_t)((unsigned char)s[*pos + 2] & 0x3f) << 6) |
              ((uint32_t)((unsigned char)s[*pos + 3] & 0x3f));
        *pos += 4;
        return 1;
    }
    *cp = c;
    (*pos)++;
    return 0;
}

static int gpt2_codepoint_to_byte(uint32_t cp, unsigned char *out)
{
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || cp >= 174)
    {
        if (cp <= 255)
        {
            *out = (unsigned char)cp;
            return 1;
        }
    }
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++)
    {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)
        {
            continue;
        }
        if (cp == 256 + n)
        {
            *out = (unsigned char)b;
            return 1;
        }
        n++;
    }
    return 0;
}

static char *token_decoded_text(qw3_engine *engine, int token, size_t *out_len)
{
    size_t raw_len = 0;
    char *raw = qw3_token_text(engine, token, &raw_len);
    if (!raw)
    {
        if (out_len)
            *out_len = 0;
        return NULL;
    }
    char *out = malloc(raw_len + 1);
    if (!out)
    {
        free(raw);
        if (out_len)
            *out_len = 0;
        return NULL;
    }
    size_t ip = 0;
    size_t op = 0;
    while (ip < raw_len)
    {
        uint32_t cp = 0;
        size_t before = ip;
        utf8_read_cp(raw, raw_len, &ip, &cp);
        unsigned char b = 0;
        if (gpt2_codepoint_to_byte(cp, &b))
        {
            out[op++] = (char)b;
        }
        else
        {
            size_t n = ip - before;
            memcpy(out + op, raw + before, n);
            op += n;
        }
    }
    out[op] = '\0';
    free(raw);
    if (out_len)
        *out_len = op;
    return out;
}

static void emit_token(void *ud, int token)
{
    emit_ctx *ctx = (emit_ctx *)ud;
    if (ctx && ctx->capture)
    {
        qw3_tokens_push(ctx->capture, token);
    }
    size_t len = 0;
    char *text = token_decoded_text(ctx->engine, token, &len);
    if (text && len)
    {
        fwrite(text, 1, len, stdout);
        fflush(stdout);
    }
    free(text);
}

static void emit_done(void *ud)
{
    (void)ud;
    fputc('\n', stdout);
    fflush(stdout);
}

static char *read_file_text(const char *path, size_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return NULL;
    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0)
    {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0)
    {
        fclose(fp);
        return NULL;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf)
    {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    if (got != (size_t)n)
    {
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    if (out_len)
        *out_len = got;
    return buf;
}

static void print_token_piece(qw3_engine *engine, int token)
{
    size_t len = 0;
    char *text = token_decoded_text(engine, token, &len);
    putchar('"');
    for (size_t i = 0; text && i < len; i++)
    {
        unsigned char c = (unsigned char)text[i];
        if (c == '\\' || c == '"')
        {
            putchar('\\');
            putchar(c);
        }
        else if (c == '\n')
        {
            fputs("\\n", stdout);
        }
        else if (c == '\r')
        {
            fputs("\\r", stdout);
        }
        else if (c == '\t')
        {
            fputs("\\t", stdout);
        }
        else if (c >= 32 && c < 127)
        {
            putchar(c);
        }
        else
        {
            printf("\\x%02x", c);
        }
    }
    putchar('"');
    free(text);
}

static void json_string(FILE *fp, const char *text, size_t len)
{
    fputc('"', fp);
    for (size_t i = 0; text && i < len; i++)
    {
        unsigned char c = (unsigned char)text[i];
        if (c == '"' || c == '\\')
        {
            fputc('\\', fp);
            fputc(c, fp);
        }
        else if (c == '\n')
        {
            fputs("\\n", fp);
        }
        else if (c == '\r')
        {
            fputs("\\r", fp);
        }
        else if (c == '\t')
        {
            fputs("\\t", fp);
        }
        else if (c < 32)
        {
            fprintf(fp, "\\u%04x", c);
        }
        else
        {
            fputc(c, fp);
        }
    }
    fputc('"', fp);
}

static void json_token(FILE *fp, qw3_engine *engine, int token)
{
    size_t len = 0;
    char *text = token_decoded_text(engine, token, &len);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    json_string(fp, text ? text : "", len);
    fprintf(fp, ",\"bytes\":[");
    for (size_t i = 0; text && i < len; i++)
    {
        if (i)
            fputc(',', fp);
        fprintf(fp, "%u", (unsigned char)text[i]);
    }
    fprintf(fp, "]}");
    free(text);
}

static int file_payload_size(FILE *fp, uint64_t *out)
{
    long cur = ftell(fp);
    if (cur < 0)
        return -1;
    if (fseek(fp, 0, SEEK_END) != 0)
        return -1;
    long end = ftell(fp);
    if (end < 0)
        return -1;
    if (fseek(fp, cur, SEEK_SET) != 0)
        return -1;
    *out = (uint64_t)end;
    return 0;
}

static int dump_logprobs(qw3_engine *engine, const qw3_tokens *prompt,
                         int ctx_size, int n_predict, int top_k,
                         sample_opts *sample, const char *path)
{
    qw3_session *session = NULL;
    char err[256] = {0};
    FILE *fp = fopen(path, "wb");
    if (!fp)
    {
        fprintf(stderr, "qw3: cannot open dump file %s\n", path);
        return -1;
    }
    if (qw3_session_create(&session, engine, ctx_size) != 0 ||
        qw3_session_sync(session, prompt, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "qw3: dump-logprobs prefill failed: %s\n", err);
        fclose(fp);
        qw3_session_free(session);
        return -1;
    }
    if (top_k <= 0)
        top_k = 20;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema\":\"qw3-local-logprobs-v1\",\n");
    fprintf(fp, "  \"ctx_size\":%d,\n", ctx_size);
    fprintf(fp, "  \"n_predict\":%d,\n", n_predict);
    fprintf(fp, "  \"temperature\":%.8g,\n", sample->temperature);
    fprintf(fp, "  \"top_k\":%d,\n", top_k);
    fprintf(fp, "  \"sample_top_k\":%d,\n", sample->sample_top_k);
    fprintf(fp, "  \"top_p\":%.8g,\n", sample->top_p);
    fprintf(fp, "  \"min_p\":%.8g,\n", sample->min_p);
    fprintf(fp, "  \"prompt_tokens\":[");
    for (int i = 0; i < prompt->len; i++)
    {
        if (i)
            fputc(',', fp);
        fprintf(fp, "%d", prompt->v[i]);
    }
    fprintf(fp, "],\n  \"steps\":[\n");

    const int eos = qw3_token_eos(engine);
    for (int step = 0; step < n_predict; step++)
    {
        qw3_token_score *scores = calloc((size_t)top_k, sizeof(*scores));
        if (!scores)
        {
            fclose(fp);
            qw3_session_free(session);
            return -1;
        }
        int n = qw3_session_top_logprobs(session, scores, top_k);
        int selected = qw3_session_sample(session, sample->temperature,
                                          sample->sample_top_k,
                                          sample->top_p, sample->min_p,
                                          &sample->rng);
        if (selected < 0)
        {
            fprintf(stderr, "qw3: dump-logprobs sampling failed\n");
            free(scores);
            fclose(fp);
            qw3_session_free(session);
            return -1;
        }
        if (step)
            fprintf(fp, ",\n");
        fprintf(fp, "    {\"step\":%d,\"selected\":", step);
        json_token(fp, engine, selected);
        fprintf(fp, ",\"top\":[");
        for (int i = 0; i < n; i++)
        {
            if (i)
                fputc(',', fp);
            fprintf(fp, "{\"token\":");
            json_token(fp, engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g}", scores[i].logit);
        }
        fprintf(fp, "]}");
        free(scores);
        if (selected == eos)
            break;
        if (qw3_session_eval(session, selected, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "qw3: dump-logprobs eval failed: %s\n", err);
            fclose(fp);
            qw3_session_free(session);
            return -1;
        }
    }
    fprintf(fp, "\n  ]\n}\n");
    fclose(fp);
    qw3_session_free(session);
    return 0;
}

static int generate_from_session(qw3_engine *engine, qw3_session *session,
                                 int n_predict, emit_ctx *emit,
                                 sample_opts *sample)
{
    char err[256] = {0};
    const int eos = qw3_token_eos(engine);
    for (int i = 0; i < n_predict; i++)
    {
        int token = qw3_session_sample(session, sample->temperature,
                                       sample->sample_top_k, sample->top_p,
                                       sample->min_p, &sample->rng);
        if (token < 0)
            return -1;
        if (token == eos)
            break;
        emit_token(emit, token);
        if (qw3_session_eval(session, token, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "qw3: generation step failed: %s\n", err);
            return -1;
        }
    }
    emit_done(emit);
    return 0;
}

#if !QW3_CLI_ENABLE_INTERNAL_TESTS
static int cli_public_option_takes_value(const char *arg)
{
    return !strcmp(arg, "-m") ||
           !strcmp(arg, "-p") ||
           !strcmp(arg, "--prompt-file") ||
           !strcmp(arg, "-sys") ||
           !strcmp(arg, "--system") ||
           !strcmp(arg, "--system-file") ||
           !strcmp(arg, "-n") ||
           !strcmp(arg, "--temp") ||
           !strcmp(arg, "--sample-top-k") ||
           !strcmp(arg, "--top-p") ||
           !strcmp(arg, "--min-p") ||
           !strcmp(arg, "--seed") ||
           !strcmp(arg, "--ctx") ||
           !strcmp(arg, "--ngl") ||
           !strcmp(arg, "-ctk") ||
           !strcmp(arg, "--ctk") ||
           !strcmp(arg, "-ctv") ||
           !strcmp(arg, "--ctv");
}

static int cli_internal_option(const char *arg)
{
    if (!arg || arg[0] != '-')
        return 0;
    if (!strncmp(arg, "--metal-", 8))
        return 1;
    return !strcmp(arg, "--inspect") ||
           !strcmp(arg, "--layer-types") ||
           !strcmp(arg, "--probe-token") ||
           !strcmp(arg, "--tokenize") ||
           !strcmp(arg, "--chat-tokenize") ||
           !strcmp(arg, "--top-k") ||
           !strcmp(arg, "--dump-logprobs") ||
           !strcmp(arg, "--logprobs-top-k") ||
           !strcmp(arg, "--session-roundtrip") ||
           !strcmp(arg, "--save-session") ||
           !strcmp(arg, "--load-session") ||
           !strcmp(arg, "--trace-layers") ||
           !strcmp(arg, "--dump-trace");
}

static int reject_internal_cli_options(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (cli_public_option_takes_value(argv[i]))
        {
            i++;
            continue;
        }
        if (cli_internal_option(argv[i]))
        {
            fprintf(stderr,
                    "qw3: internal diagnostic/test option '%s' is disabled in the client build\n"
                    "qw3: rebuild with -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 to enable developer diagnostics\n",
                    argv[i]);
            return -1;
        }
    }
    return 0;
}
#endif

static void append_generated_assistant(qw3_engine *engine, qw3_tokens *transcript,
                                       const qw3_tokens *generated)
{
    for (int i = 0; i < generated->len; i++)
    {
        qw3_tokens_push(transcript, generated->v[i]);
    }
    int eos = qw3_token_eos(engine);
    if (eos >= 0 && (transcript->len == 0 ||
                     transcript->v[transcript->len - 1] != eos))
    {
        qw3_tokens_push(transcript, eos);
    }
}

static int generate_chat_turn(qw3_engine *engine, qw3_backend backend,
                              qw3_session *session, qw3_tokens *transcript,
                              int ctx_size, int n_predict,
                              qw3_think_mode think_mode, sample_opts *sample)
{
    char err[256] = {0};
    qw3_tokens generated = {0};
    emit_ctx emit = {
        .engine = engine,
        .capture = &generated,
    };

    qw3_chat_append_assistant_prefix(engine, transcript, think_mode);

    int rc = -1;
    if (backend == QW3_BACKEND_METAL)
    {
        if (sample->temperature > 0.0f)
        {
            rc = qw3_engine_metal_generate_sample(
                engine, transcript, n_predict, ctx_size,
                sample->temperature, sample->sample_top_k,
                sample->top_p, sample->min_p, &sample->rng,
                emit_token, emit_done, &emit);
        }
        else
        {
            rc = qw3_engine_metal_generate_argmax(
                engine, transcript, n_predict, ctx_size,
                emit_token, emit_done, &emit);
        }
    }
    else if (qw3_session_sync(session, transcript, err, sizeof(err)) == 0)
    {
        rc = generate_from_session(engine, session, n_predict, &emit, sample);
    }
    else
    {
        fprintf(stderr, "qw3: chat prefill failed: %s\n", err);
    }

    if (rc == 0)
    {
        append_generated_assistant(engine, transcript, &generated);
    }
    qw3_tokens_free(&generated);
    return rc;
}

static void interactive_help(void)
{
    fprintf(stderr,
            "Commands:\n"
            "  /help              Show this help\n"
            "  /read PATH         Send a file as the next user message\n"
            "  /ctx               Print current token count and context size\n"
            "  /new               Start a new conversation\n"
            "  /think             Enable thinking mode for later turns\n"
            "  /nothink           Disable thinking mode for later turns\n"
            "  /quit              Exit\n");
}

static int interactive_chat(qw3_engine *engine, qw3_backend backend,
                            const char *system_prompt, int ctx_size,
                            int n_predict, qw3_think_mode think_mode,
                            sample_opts *sample)
{
    qw3_session *session = NULL;
    if (backend != QW3_BACKEND_METAL &&
        qw3_session_create(&session, engine, ctx_size) != 0)
    {
        fprintf(stderr, "qw3: cannot create CPU chat session\n");
        return 1;
    }

    qw3_tokens transcript = {0};
    if (system_prompt && system_prompt[0])
    {
        qw3_chat_append_message(engine, &transcript, "system", system_prompt);
    }

    fprintf(stderr, "qw3 chat ready. Type /help for commands.\n");
    char *line = NULL;
    size_t cap = 0;
    int rc = 0;
    for (;;)
    {
        if (isatty(STDIN_FILENO))
        {
            fprintf(stderr, "\nqw3> ");
            fflush(stderr);
        }
        ssize_t nread = getline(&line, &cap, stdin);
        if (nread < 0)
            break;
        while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r'))
        {
            line[--nread] = '\0';
        }
        if (nread == 0)
            continue;

        char *message = line;
        char *owned_message = NULL;
        if (line[0] == '/')
        {
            if (!strcmp(line, "/quit") || !strcmp(line, "/exit"))
            {
                break;
            }
            else if (!strcmp(line, "/help"))
            {
                interactive_help();
                continue;
            }
            else if (!strcmp(line, "/ctx"))
            {
                fprintf(stderr, "tokens=%d ctx=%d\n", transcript.len, ctx_size);
                continue;
            }
            else if (!strcmp(line, "/new"))
            {
                qw3_tokens_free(&transcript);
                memset(&transcript, 0, sizeof(transcript));
                if (system_prompt && system_prompt[0])
                {
                    qw3_chat_append_message(engine, &transcript, "system", system_prompt);
                }
                if (session)
                    qw3_session_invalidate(session);
                fprintf(stderr, "new conversation\n");
                continue;
            }
            else if (!strcmp(line, "/think"))
            {
                think_mode = QW3_THINK_ON;
                fprintf(stderr, "thinking enabled\n");
                continue;
            }
            else if (!strcmp(line, "/nothink"))
            {
                think_mode = QW3_THINK_NONE;
                fprintf(stderr, "thinking disabled\n");
                continue;
            }
            else if (!strncmp(line, "/read ", 6))
            {
                errno = 0;
                owned_message = read_file_text(line + 6, NULL);
                if (!owned_message)
                {
                    fprintf(stderr, "qw3: cannot read %s: %s\n",
                            line + 6, errno ? strerror(errno) : "read failed");
                    continue;
                }
                message = owned_message;
            }
            else
            {
                fprintf(stderr, "unknown command: %s\n", line);
                continue;
            }
        }

        qw3_chat_append_message(engine, &transcript, "user", message);
        free(owned_message);
        if (transcript.len >= ctx_size)
        {
            fprintf(stderr, "qw3: context is full (%d/%d tokens)\n",
                    transcript.len, ctx_size);
            rc = 1;
            break;
        }
        if (generate_chat_turn(engine, backend, session, &transcript,
                               ctx_size, n_predict, think_mode, sample) != 0)
        {
            rc = 1;
            break;
        }
    }
    free(line);
    qw3_tokens_free(&transcript);
    qw3_session_free(session);
    return rc;
}

static void usage(void)
{
    fprintf(stderr,
            "qw3 — Qwen3.6-35B-A3B inference engine\n"
            "\n"
            "Usage:\n"
            "  qw3 [options]\n"
            "\n"
            "Options:\n"
            "  -m PATH     Model GGUF path (default: ./qw3.gguf)\n"
            "  -p TEXT      One-shot prompt\n"
            "  --prompt-file PATH\n"
            "              Read one-shot prompt from a file\n"
            "  -sys TEXT    System prompt\n"
            "  --system-file PATH\n"
            "              Read system prompt from a file\n"
            "  -n N         Max tokens to generate (default: 512)\n"
            "  --temp N     Sampling temperature (default: 0, greedy)\n"
            "  --sample-top-k N\n"
            "              Sampling top-k (default: 40, 0 = full vocab)\n"
            "  --top-p N    Sampling nucleus top-p (default: 0.95)\n"
            "  --min-p N    Sampling min-p relative floor (default: 0)\n"
            "  --seed N     Sampling seed (default: fixed)\n"
            "  --ctx N      Context size (default: 32768)\n"
            "  --ngl N      Metal layers to keep on GPU, 0..40 (default: 40)\n"
            "  -ctk TYPE    Metal K cache type: f32, f16, or q8_0 (use with -ctv)\n"
            "  -ctv TYPE    Metal V cache type: f32, f16, or q8_0 (use with -ctk)\n"
            "  --kv-f16     Use f16 Metal GQA KV cache (recommended for large ctx)\n"
            "  --kv-f32     Use f32 Metal GQA KV cache\n"
            "  --kv-q8      Use q8_0 Metal GQA KV cache (experimental)\n"
            "  --cpu        Use CPU backend\n"
            "  --metal      Use Metal backend when compiled in\n"
            "  --nothink    Disable thinking mode\n"
#if QW3_CLI_ENABLE_INTERNAL_TESTS
            "  --inspect    Print GGUF metadata summary after loading\n"
            "  --layer-types N\n"
            "              Print tensor quantization types for one layer\n"
            "  --probe-token ID\n"
            "              Run a small CPU probe: embedding, layer-0 norm, MoE router\n"
            "  --tokenize   Tokenize -p TEXT and print token IDs\n"
            "  --chat-tokenize\n"
            "              Tokenize -p TEXT as a single user chat prompt\n"
            "  --top-k N    Prefill -p TEXT and print top-N logits without sampling\n"
            "  --dump-logprobs PATH\n"
            "              Prefill -p TEXT and dump per-step selected token + top logits JSON\n"
            "  --logprobs-top-k N\n"
            "              Number of top logits to store in --dump-logprobs (default: 20)\n"
            "  --session-roundtrip\n"
            "              Prefill -p TEXT, save/load session payload, compare top-5\n"
            "  --save-session PATH\n"
            "              Prefill -p TEXT and save the CPU session payload\n"
            "  --load-session PATH\n"
            "              Load a saved CPU session payload and continue generation\n"
            "  --trace-layers\n"
            "              Trace final prompt token through all layers (CPU reference)\n"
            "  --dump-trace PATH\n"
            "              Dump final prompt token layer trace as JSON\n"
            "  --metal-rmsnorm-test\n"
            "              Run a tiny Metal RMSNorm kernel and compare with CPU\n"
            "  --metal-rmsnorm-weight-test ID\n"
            "              Run weighted RMSNorm on a real token embedding and compare with CPU\n"
            "  --metal-embed-test ID\n"
            "              Dequantize one q8_0 embedding row on Metal and compare with CPU\n"
            "  --metal-matvec-q8-test ID\n"
            "              Run blk.0 attn_qkv q8_0 matvec on Metal and compare with CPU\n"
            "  --metal-deltanet-proj-test ID\n"
            "              Run layer-0 DeltaNet qkv/z projections on Metal and compare with CPU\n"
            "  --metal-deltanet-conv-test ID\n"
            "              Run layer-0 DeltaNet zero-state conv1d on Metal and compare with CPU\n"
            "  --metal-deltanet-conv-step-test ID\n"
            "              Run layer-0 DeltaNet non-zero-state conv1d on Metal\n"
            "  --metal-deltanet-l2-test ID\n"
            "              Run layer-0 DeltaNet Q/K per-head L2Norm on Metal and compare with CPU\n"
            "  --metal-deltanet-gates-test ID\n"
            "              Run layer-0 DeltaNet alpha/beta f32 gates on Metal and compare with CPU\n"
            "  --metal-deltanet-recur-zero-test ID\n"
            "              Run layer-0 DeltaNet zero-state recurrence on Metal and compare with CPU\n"
            "  --metal-deltanet-recur-test ID\n"
            "              Run layer-0 DeltaNet non-zero-state recurrence on Metal and compare with CPU\n"
            "  --metal-deltanet-recur-step-test ID\n"
            "              Run layer-0 DeltaNet conv-step/L2/recur chain on Metal\n"
            "  --metal-deltanet-gated-norm-test ID\n"
            "              Run layer-0 DeltaNet gated RMSNorm on Metal and compare with CPU\n"
            "  --metal-deltanet-out-test ID\n"
            "              Run layer-0 DeltaNet ssm_out projection on Metal and compare with CPU\n"
            "  --metal-deltanet-branch-step-test ID\n"
            "              Run layer-0 stateful DeltaNet attention branch on Metal\n"
            "  --metal-deltanet-layer-step-test ID\n"
            "              Run layer-0 stateful DeltaNet layer output on Metal\n"
            "  --metal-deltanet-layer2-test ID\n"
            "              Run layer-0 stateful DeltaNet layer for two sequential tokens\n"
            "  --metal-deltanet-layer4-test ID\n"
            "              Run layer-0 stateful DeltaNet layer for four sequential tokens\n"
            "  --metal-deltanet-layer8-test ID\n"
            "              Run layer-0 stateful DeltaNet layer for eight sequential tokens\n"
            "  --metal-deltanet-branch-test ID\n"
            "              Run composed layer-0 DeltaNet attention branch on Metal and compare with CPU\n"
            "  --metal-deltanet-resid-norm-test ID\n"
            "              Run layer-0 residual plus ffn_norm on Metal and compare with CPU\n"
            "  --metal-moe-router-test ID\n"
            "              Run layer-0 MoE router matvec/top-8 on Metal and compare with CPU\n"
            "  --metal-moe-shared-test ID\n"
            "              Run layer-0 MoE shared expert on Metal and compare with CPU\n"
            "  --metal-moe-iq4-down-test ID\n"
            "              Run layer-0 sparse expert IQ4_XS down matvec on Metal and compare with CPU\n"
            "  --metal-moe-q6-down-test ID\n"
            "              Run layer-34 sparse expert Q6_K down matvec on Metal and compare with CPU\n"
            "  --metal-moe-iq3-test ID\n"
            "              Run layer-0 sparse expert IQ3_S gate/up matvecs on Metal and compare with CPU\n"
            "  --metal-moe-sparse-top1-test ID\n"
            "              Run layer-0 sparse expert top-1 path on Metal and compare with CPU\n"
            "  --metal-moe-sparse-top8-test ID\n"
            "              Run layer-0 sparse expert top-8 weighted path on Metal and compare with CPU\n"
            "  --metal-moe-layer-test ID\n"
            "              Run layer-0 MoE sparse+shared path on Metal and compare with CPU\n"
            "  --metal-moe-real-layer-test ID\n"
            "              Run layer-0 DeltaNet ffn_norm plus MoE on Metal and compare with CPU\n"
            "  --metal-deltanet3-test ID\n"
            "              Run layer-0..2 DeltaNet+MoE sequence on Metal and compare with CPU\n"
            "  --metal-mixed4-test ID\n"
            "              Run layer-0..3 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed8-test ID\n"
            "              Run layer-0..7 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed16-test ID\n"
            "              Run layer-0..15 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed32-test ID\n"
            "              Run layer-0..31 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed33-test ID\n"
            "              Run layer-0..32 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed34-test ID\n"
            "              Run layer-0..33 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed35-test ID\n"
            "              Run layer-0..34 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed36-test ID\n"
            "              Run layer-0..35 mixed DeltaNet/GQA sequence on Metal and compare with CPU\n"
            "  --metal-mixed40-test ID\n"
            "              Run all 40 layers as a first-token mixed Metal diagnostic\n"
            "  --metal-logits-test ID\n"
            "              Run all 40 layers plus output norm/lm_head on Metal\n"
            "  --metal-decode-test\n"
            "              Prefill -p TEXT with stateful Metal layers and compare logits\n"
            "  --metal-session-decode-test\n"
            "              Prefill -p TEXT with persistent Metal session buffers and compare logits\n"
            "  --metal-session-test\n"
            "              Allocate and clear persistent Metal session buffers\n"
            "  --metal-session-embed-test ID\n"
            "              Embed one token into persistent Metal session x0 and compare\n"
            "  --metal-session-rmsnorm-test ID\n"
            "              Run session x0 embedding then layer-0 attn_norm into x1\n"
            "  --metal-session-qkv-test ID\n"
            "              Run session x1 through layer-0 qkv projection into scratch\n"
            "  --metal-session-prefill-q8-batch-test ID\n"
            "              Run 4-token batched q8 prefill projection and compare\n"
            "  --metal-session-gqa-prefill-batch-test ID\n"
            "              Run 4-token batched GQA prefill attention and compare\n"
            "  --metal-session-z-test ID\n"
            "              Run layer-0 gate projection into scratch after qkv\n"
            "  --metal-session-conv-test ID\n"
            "              Run session qkv through layer-0 zero-state conv1d\n"
            "  --metal-session-l2norm-test ID\n"
            "              Run session conv output through Q/K L2Norm heads\n"
            "  --metal-session-gates-test ID\n"
            "              Run session layer-0 SSM alpha/beta f32 projections\n"
            "  --metal-session-recur-zero-test ID\n"
            "              Run session zero-state DeltaNet recurrence from persistent buffers\n"
            "  --metal-session-recur-step-test ID\n"
            "              Run session persistent DeltaNet recurrence over two tokens\n"
            "  --metal-session-gated-rmsnorm-test ID\n"
            "              Run session DeltaNet gated RMSNorm from persistent core and z\n"
            "  --metal-session-attn-out-test ID\n"
            "              Run session DeltaNet inner through layer-0 ssm_out into x1\n"
            "  --metal-session-ffn-norm-test ID\n"
            "              Run session residual + layer-0 ffn_norm after ssm_out\n"
            "  --metal-session-layer0-test ID\n"
            "              Run full layer-0 through session buffers and compare\n"
            "  --metal-session-gqa-project-test ID\n"
            "              Run session layer-3 GQA projection, RoPE and KV-cache write\n"
            "  --metal-session-gqa-single-test ID\n"
            "              Run session layer-3 single-token GQA through attn_o\n"
            "  --metal-session-gqa-cached2-test ID\n"
            "              Run session layer-3 GQA over two cached tokens through attn_o\n"
            "  --metal-session-gqa-cached-bench ID N\n"
            "              Benchmark session layer-3 cached attention at N cached tokens\n"
            "  --metal-greedy-test N\n"
            "              Compare N greedy decode steps using Metal logits\n"
            "  --metal-run N\n"
            "              Generate N greedy token IDs using Metal logits only\n"
            "  --metal-run-quiet\n"
            "              Suppress per-token Metal run output and print timing summary\n"
            "  --metal-gqa-project-test ID\n"
            "              Run layer-3 GQA q/k/v projection, Q/K norm and RoPE on Metal\n"
            "  --metal-gqa-single-test ID\n"
            "              Run layer-3 GQA single-token attention plus attn_o on Metal\n"
            "  --metal-gqa-attend2-test ID\n"
            "              Run layer-3 GQA two-token attention inner on Metal\n"
            "  --metal-gqa-attend4-test ID\n"
            "              Run layer-3 GQA four-token attention inner on Metal\n"
            "  --metal-gqa-branch4-test ID\n"
            "              Run layer-3 GQA four-token projection/cache/attn_o branch on Metal\n"
            "  --metal-gqa-layer4-test ID\n"
            "              Run layer-3 GQA four-token final layer output on Metal\n"
            "  --metal-gqa-real-layer-test ID\n"
            "              Run layer-3 GQA residual plus MoE on Metal and compare with CPU\n"
#endif
            "  --help       Show this help\n");
}

int main(int argc, char **argv)
{
    const char *model_path = "./qw3.gguf";
    const char *prompt = NULL;
    char *prompt_owned = NULL;
    const char *system_prompt = NULL;
    char *system_owned = NULL;
    int n_predict = 512;
    int ctx_size = 32768;
    int ngl = -1;
    int ngl_set = 0;
    const char *cache_type_k = NULL;
    const char *cache_type_v = NULL;
    const char *cache_type_alias = NULL;
    sample_opts sample = {
        .temperature = 0.0f,
        .sample_top_k = 40,
        .top_p = 0.95f,
        .min_p = 0.0f,
        .rng = 0x123456789abcdef0ull,
    };
#ifdef QW3_NO_METAL
    qw3_backend backend = QW3_BACKEND_CPU;
#else
    qw3_backend backend = QW3_BACKEND_METAL;
#endif
    qw3_think_mode think_mode = QW3_THINK_ON;
    int inspect = 0;
    int layer_types = -1;
    int tokenize = 0;
    int chat_tokenize = 0;
    int probe_token = -1;
    int top_k_logits = 0;
    const char *dump_logprobs_path = NULL;
    int logprobs_top_k = 20;
    int session_roundtrip = 0;
    int trace_layers = 0;
    const char *dump_trace_path = NULL;
    int metal_rmsnorm_test = 0;
    int metal_rmsnorm_weight_test = -1;
    int metal_embed_test = -1;
    int metal_matvec_q8_test = -1;
    int metal_deltanet_proj_test = -1;
    int metal_deltanet_conv_test = -1;
    int metal_deltanet_conv_step_test = -1;
    int metal_deltanet_l2_test = -1;
    int metal_deltanet_gates_test = -1;
    int metal_deltanet_recur_zero_test = -1;
    int metal_deltanet_recur_test = -1;
    int metal_deltanet_recur_step_test = -1;
    int metal_deltanet_gated_norm_test = -1;
    int metal_deltanet_out_test = -1;
    int metal_deltanet_branch_step_test = -1;
    int metal_deltanet_layer_step_test = -1;
    int metal_deltanet_layer2_test = -1;
    int metal_deltanet_layer4_test = -1;
    int metal_deltanet_layer8_test = -1;
    int metal_deltanet_branch_test = -1;
    int metal_deltanet_resid_norm_test = -1;
    int metal_moe_router_test = -1;
    int metal_moe_shared_test = -1;
    int metal_moe_iq4_down_test = -1;
    int metal_moe_q6_down_test = -1;
    int metal_moe_iq3_test = -1;
    int metal_moe_sparse_top1_test = -1;
    int metal_moe_sparse_top8_test = -1;
    int metal_moe_layer_test = -1;
    int metal_moe_real_layer_test = -1;
    int metal_deltanet3_test = -1;
    int metal_mixed4_test = -1;
    int metal_mixed8_test = -1;
    int metal_mixed16_test = -1;
    int metal_mixed32_test = -1;
    int metal_mixed33_test = -1;
    int metal_mixed34_test = -1;
    int metal_mixed35_test = -1;
    int metal_mixed36_test = -1;
    int metal_mixed40_test = -1;
    int metal_logits_test = -1;
    int metal_decode_test = 0;
    int metal_session_decode_test = 0;
    int metal_session_test = 0;
    int metal_session_embed_test = -1;
    int metal_session_rmsnorm_test = -1;
    int metal_session_qkv_test = -1;
    int metal_session_prefill_q8_batch_test = -1;
    int metal_session_gqa_prefill_batch_test = -1;
    int metal_session_z_test = -1;
    int metal_session_conv_test = -1;
    int metal_session_l2norm_test = -1;
    int metal_session_gates_test = -1;
    int metal_session_recur_zero_test = -1;
    int metal_session_recur_step_test = -1;
    int metal_session_gated_rmsnorm_test = -1;
    int metal_session_attn_out_test = -1;
    int metal_session_ffn_norm_test = -1;
    int metal_session_layer0_test = -1;
    int metal_session_gqa_project_test = -1;
    int metal_session_gqa_single_test = -1;
    int metal_session_gqa_cached2_test = -1;
    int metal_session_gqa_cached_bench_token = -1;
    int metal_session_gqa_cached_bench_n = 0;
    int metal_greedy_test = 0;
    int metal_run = 0;
    int metal_run_quiet = 0;
    int metal_gqa_project_test = -1;
    int metal_gqa_single_test = -1;
    int metal_gqa_attend2_test = -1;
    int metal_gqa_attend4_test = -1;
    int metal_gqa_branch4_test = -1;
    int metal_gqa_layer4_test = -1;
    int metal_gqa_real_layer_test = -1;
    const char *save_session_path = NULL;
    const char *load_session_path = NULL;

#if !QW3_CLI_ENABLE_INTERNAL_TESTS
    if (reject_internal_cli_options(argc, argv) != 0)
    {
        free(prompt_owned);
        free(system_owned);
        return 1;
    }
#endif

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            usage();
            return 0;
        }
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
        {
            model_path = argv[++i];
        }
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc)
        {
            prompt = argv[++i];
        }
        else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc)
        {
            const char *path = argv[++i];
            free(prompt_owned);
            errno = 0;
            prompt_owned = read_file_text(path, NULL);
            if (!prompt_owned)
            {
                fprintf(stderr, "qw3: cannot read prompt file %s: %s\n",
                        path, errno ? strerror(errno) : "read failed");
                free(system_owned);
                return 1;
            }
            prompt = prompt_owned;
        }
        else if ((strcmp(argv[i], "-sys") == 0 ||
                  strcmp(argv[i], "--system") == 0) &&
                 i + 1 < argc)
        {
            system_prompt = argv[++i];
        }
        else if (strcmp(argv[i], "--system-file") == 0 && i + 1 < argc)
        {
            const char *path = argv[++i];
            free(system_owned);
            errno = 0;
            system_owned = read_file_text(path, NULL);
            if (!system_owned)
            {
                fprintf(stderr, "qw3: cannot read system file %s: %s\n",
                        path, errno ? strerror(errno) : "read failed");
                free(prompt_owned);
                return 1;
            }
            system_prompt = system_owned;
        }
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            n_predict = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
        {
            sample.temperature = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--sample-top-k") == 0 && i + 1 < argc)
        {
            sample.sample_top_k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
        {
            sample.top_p = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc)
        {
            sample.min_p = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            sample.rng = strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc)
        {
            ctx_size = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--ngl") == 0 && i + 1 < argc)
        {
            ngl = atoi(argv[++i]);
            ngl_set = 1;
            backend = QW3_BACKEND_METAL;
        }
        else if ((strcmp(argv[i], "-ctk") == 0 ||
                  strcmp(argv[i], "--ctk") == 0) &&
                 i + 1 < argc)
        {
            cache_type_k = argv[++i];
            backend = QW3_BACKEND_METAL;
        }
        else if ((strcmp(argv[i], "-ctv") == 0 ||
                  strcmp(argv[i], "--ctv") == 0) &&
                 i + 1 < argc)
        {
            cache_type_v = argv[++i];
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--kv-f16") == 0)
        {
            cache_type_alias = "f16";
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--kv-f32") == 0)
        {
            cache_type_alias = "f32";
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--kv-q8") == 0)
        {
            cache_type_alias = "q8_0";
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--cpu") == 0)
        {
            backend = QW3_BACKEND_CPU;
        }
        else if (strcmp(argv[i], "--metal") == 0)
        {
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--nothink") == 0)
        {
            think_mode = QW3_THINK_NONE;
        }
        else if (strcmp(argv[i], "--inspect") == 0)
        {
            inspect = 1;
        }
        else if (strcmp(argv[i], "--layer-types") == 0 && i + 1 < argc)
        {
            layer_types = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--probe-token") == 0 && i + 1 < argc)
        {
            probe_token = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tokenize") == 0)
        {
            tokenize = 1;
        }
        else if (strcmp(argv[i], "--chat-tokenize") == 0)
        {
            chat_tokenize = 1;
        }
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
        {
            top_k_logits = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--dump-logprobs") == 0 && i + 1 < argc)
        {
            dump_logprobs_path = argv[++i];
        }
        else if (strcmp(argv[i], "--logprobs-top-k") == 0 && i + 1 < argc)
        {
            logprobs_top_k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--session-roundtrip") == 0)
        {
            session_roundtrip = 1;
        }
        else if (strcmp(argv[i], "--save-session") == 0 && i + 1 < argc)
        {
            save_session_path = argv[++i];
        }
        else if (strcmp(argv[i], "--load-session") == 0 && i + 1 < argc)
        {
            load_session_path = argv[++i];
        }
        else if (strcmp(argv[i], "--trace-layers") == 0)
        {
            trace_layers = 1;
        }
        else if (strcmp(argv[i], "--dump-trace") == 0 && i + 1 < argc)
        {
            dump_trace_path = argv[++i];
        }
        else if (strcmp(argv[i], "--metal-rmsnorm-test") == 0)
        {
            metal_rmsnorm_test = 1;
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-rmsnorm-weight-test") == 0 && i + 1 < argc)
        {
            metal_rmsnorm_weight_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-embed-test") == 0 && i + 1 < argc)
        {
            metal_embed_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-matvec-q8-test") == 0 && i + 1 < argc)
        {
            metal_matvec_q8_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-proj-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_proj_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-conv-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_conv_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-conv-step-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_conv_step_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-l2-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_l2_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-gates-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_gates_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-recur-zero-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_recur_zero_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-recur-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_recur_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-recur-step-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_recur_step_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-gated-norm-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_gated_norm_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-out-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_out_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-branch-step-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_branch_step_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-layer-step-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_layer_step_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-layer2-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_layer2_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-layer4-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_layer4_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-layer8-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_layer8_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-branch-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_branch_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet-resid-norm-test") == 0 && i + 1 < argc)
        {
            metal_deltanet_resid_norm_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-router-test") == 0 && i + 1 < argc)
        {
            metal_moe_router_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-shared-test") == 0 && i + 1 < argc)
        {
            metal_moe_shared_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-iq4-down-test") == 0 && i + 1 < argc)
        {
            metal_moe_iq4_down_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-q6-down-test") == 0 && i + 1 < argc)
        {
            metal_moe_q6_down_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-iq3-test") == 0 && i + 1 < argc)
        {
            metal_moe_iq3_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-sparse-top1-test") == 0 && i + 1 < argc)
        {
            metal_moe_sparse_top1_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-sparse-top8-test") == 0 && i + 1 < argc)
        {
            metal_moe_sparse_top8_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-layer-test") == 0 && i + 1 < argc)
        {
            metal_moe_layer_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-moe-real-layer-test") == 0 && i + 1 < argc)
        {
            metal_moe_real_layer_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-deltanet3-test") == 0 && i + 1 < argc)
        {
            metal_deltanet3_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed4-test") == 0 && i + 1 < argc)
        {
            metal_mixed4_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed8-test") == 0 && i + 1 < argc)
        {
            metal_mixed8_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed16-test") == 0 && i + 1 < argc)
        {
            metal_mixed16_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed32-test") == 0 && i + 1 < argc)
        {
            metal_mixed32_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed33-test") == 0 && i + 1 < argc)
        {
            metal_mixed33_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed34-test") == 0 && i + 1 < argc)
        {
            metal_mixed34_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed35-test") == 0 && i + 1 < argc)
        {
            metal_mixed35_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed36-test") == 0 && i + 1 < argc)
        {
            metal_mixed36_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-mixed40-test") == 0 && i + 1 < argc)
        {
            metal_mixed40_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-logits-test") == 0 && i + 1 < argc)
        {
            metal_logits_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-decode-test") == 0)
        {
            metal_decode_test = 1;
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-decode-test") == 0)
        {
            metal_session_decode_test = 1;
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-test") == 0)
        {
            metal_session_test = 1;
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-embed-test") == 0 && i + 1 < argc)
        {
            metal_session_embed_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-rmsnorm-test") == 0 && i + 1 < argc)
        {
            metal_session_rmsnorm_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-qkv-test") == 0 && i + 1 < argc)
        {
            metal_session_qkv_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-prefill-q8-batch-test") == 0 && i + 1 < argc)
        {
            metal_session_prefill_q8_batch_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-gqa-prefill-batch-test") == 0 && i + 1 < argc)
        {
            metal_session_gqa_prefill_batch_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-z-test") == 0 && i + 1 < argc)
        {
            metal_session_z_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-conv-test") == 0 && i + 1 < argc)
        {
            metal_session_conv_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-l2norm-test") == 0 && i + 1 < argc)
        {
            metal_session_l2norm_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-gates-test") == 0 && i + 1 < argc)
        {
            metal_session_gates_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-recur-zero-test") == 0 && i + 1 < argc)
        {
            metal_session_recur_zero_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-recur-step-test") == 0 && i + 1 < argc)
        {
            metal_session_recur_step_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-gated-rmsnorm-test") == 0 && i + 1 < argc)
        {
            metal_session_gated_rmsnorm_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-attn-out-test") == 0 && i + 1 < argc)
        {
            metal_session_attn_out_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-ffn-norm-test") == 0 && i + 1 < argc)
        {
            metal_session_ffn_norm_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-layer0-test") == 0 && i + 1 < argc)
        {
            metal_session_layer0_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-gqa-project-test") == 0 && i + 1 < argc)
        {
            metal_session_gqa_project_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-gqa-single-test") == 0 && i + 1 < argc)
        {
            metal_session_gqa_single_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-gqa-cached2-test") == 0 && i + 1 < argc)
        {
            metal_session_gqa_cached2_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-session-gqa-cached-bench") == 0 && i + 2 < argc)
        {
            metal_session_gqa_cached_bench_token = atoi(argv[++i]);
            metal_session_gqa_cached_bench_n = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-greedy-test") == 0 && i + 1 < argc)
        {
            metal_greedy_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-run") == 0 && i + 1 < argc)
        {
            metal_run = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-run-quiet") == 0)
        {
            metal_run_quiet = 1;
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-gqa-project-test") == 0 && i + 1 < argc)
        {
            metal_gqa_project_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-gqa-single-test") == 0 && i + 1 < argc)
        {
            metal_gqa_single_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-gqa-attend2-test") == 0 && i + 1 < argc)
        {
            metal_gqa_attend2_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-gqa-attend4-test") == 0 && i + 1 < argc)
        {
            metal_gqa_attend4_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-gqa-branch4-test") == 0 && i + 1 < argc)
        {
            metal_gqa_branch4_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-gqa-layer4-test") == 0 && i + 1 < argc)
        {
            metal_gqa_layer4_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else if (strcmp(argv[i], "--metal-gqa-real-layer-test") == 0 && i + 1 < argc)
        {
            metal_gqa_real_layer_test = atoi(argv[++i]);
            backend = QW3_BACKEND_METAL;
        }
        else
        {
            fprintf(stderr, "qw3: unknown option '%s'\n", argv[i]);
            usage();
            return 1;
        }
    }

    if (!qw3_backend_supported(backend))
    {
        fprintf(stderr, "qw3: backend %s is not supported by this binary\n",
                qw3_backend_name(backend));
        return 1;
    }
    if (ngl_set)
    {
        if (backend != QW3_BACKEND_METAL)
        {
            fprintf(stderr, "qw3: --ngl is available only with the Metal backend\n");
            return 1;
        }
        if (ngl < 0 || ngl > QW3_CLI_N_LAYER)
        {
            fprintf(stderr, "qw3: --ngl must be in the range 0..%d\n",
                    QW3_CLI_N_LAYER);
            return 1;
        }
        char ngl_env[16];
        snprintf(ngl_env, sizeof(ngl_env), "%d", ngl);
        setenv("QW3_METAL_NGL", ngl_env, 1);
    }
    if (cache_type_alias)
    {
        if (cache_type_k || cache_type_v)
        {
            fprintf(stderr,
                    "qw3: use either --kv-f16/--kv-f32/--kv-q8 or -ctk/-ctv, not both\n");
            return 1;
        }
        cache_type_k = cache_type_alias;
        cache_type_v = cache_type_alias;
    }
    if (cache_type_k || cache_type_v)
    {
        if (backend != QW3_BACKEND_METAL)
        {
            fprintf(stderr, "qw3: -ctk/-ctv are available only with the Metal backend\n");
            return 1;
        }
        if (!cache_type_k || !cache_type_v ||
            strcmp(cache_type_k, cache_type_v) != 0)
        {
            fprintf(stderr,
                    "qw3: the Metal backend currently requires matching -ctk and -ctv types\n");
            return 1;
        }
        if (strcmp(cache_type_k, "q8_0") == 0)
        {
            setenv("QW3_METAL_KV_Q8_0", "1", 1);
            setenv("QW3_METAL_KV_F16", "0", 1);
        }
        else if (strcmp(cache_type_k, "f16") == 0)
        {
            setenv("QW3_METAL_KV_Q8_0", "0", 1);
            setenv("QW3_METAL_KV_F16", "1", 1);
        }
        else if (strcmp(cache_type_k, "f32") == 0)
        {
            setenv("QW3_METAL_KV_Q8_0", "0", 1);
            setenv("QW3_METAL_KV_F16", "0", 1);
        }
        else
        {
            fprintf(stderr,
                    "qw3: unsupported Metal KV cache type '%s' (expected f32, f16, or q8_0)\n",
                    cache_type_k);
            return 1;
        }
    }
    if (backend == QW3_BACKEND_METAL &&
        (save_session_path || load_session_path || session_roundtrip))
    {
        fprintf(stderr,
                "qw3: session save/load/roundtrip currently use CPU session state; pass --cpu for these commands\n");
        return 1;
    }

    /* Print model shape summary. */
    qw3_log(stderr, QW3_LOG_OK,
            "qw3: Qwen3.6-35B-A3B engine (Phase 3 CPU reference)\n");
    qw3_log(stderr, QW3_LOG_OK,
            "qw3: model=%s  backend=%s  ctx=%d  think=%s\n",
            model_path, qw3_backend_name(backend),
            ctx_size, qw3_think_mode_name(think_mode));

    /* Memory estimate. */
    qw3_context_memory mem = qw3_context_memory_estimate(backend, ctx_size);
    qw3_log(stderr, QW3_LOG_TIMING,
            "qw3: memory estimate: total=%.1f MB "
            "(gqa_kv=%.1f MB  deltanet=%.1f MB  scratch=%.1f MB)\n",
            (double)mem.total_bytes / (1024.0 * 1024.0),
            (double)mem.gqa_kv_bytes / (1024.0 * 1024.0),
            (double)mem.deltanet_state_bytes / (1024.0 * 1024.0),
            (double)mem.scratch_bytes / (1024.0 * 1024.0));

    qw3_engine_options opt = {
        .model_path = model_path,
        .backend = backend,
        .n_threads = 0,
        .warm_weights = false,
    };

    qw3_engine *engine = NULL;
    int rc = qw3_engine_open(&engine, &opt);
    if (rc != 0)
    {
        qw3_log(stderr, QW3_LOG_ERROR, "qw3: engine open failed\n");
        return 1;
    }

    if (inspect)
    {
        qw3_engine_inspect(engine, stdout);
    }
    if (layer_types >= 0)
    {
        qw3_engine_layer_types(engine, layer_types, stdout);
    }
    if (probe_token >= 0)
    {
        qw3_engine_probe_token(engine, probe_token, stdout);
    }
    if (tokenize)
    {
        qw3_tokens tokens = {0};
        qw3_tokenize_text(engine, prompt ? prompt : "", &tokens);
        printf("[");
        for (int i = 0; i < tokens.len; i++)
        {
            if (i)
                printf(", ");
            printf("%d", tokens.v[i]);
        }
        printf("]\n");
        qw3_tokens_free(&tokens);
    }
    if (chat_tokenize)
    {
        qw3_tokens tokens = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt ? prompt : "",
                               think_mode, &tokens);
        printf("[");
        for (int i = 0; i < tokens.len; i++)
        {
            if (i)
                printf(", ");
            printf("%d", tokens.v[i]);
        }
        printf("]\n");
        qw3_tokens_free(&tokens);
    }
    if (top_k_logits > 0)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --top-k requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_session *session = NULL;
        char err[256] = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        if (qw3_session_create(&session, engine, ctx_size) != 0 ||
            qw3_session_sync(session, &tokens, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "qw3: top-k prefill failed: %s\n", err);
            qw3_session_free(session);
            qw3_tokens_free(&tokens);
            qw3_engine_close(engine);
            return 1;
        }
        qw3_token_score *scores = calloc((size_t)top_k_logits, sizeof(*scores));
        if (!scores)
        {
            fprintf(stderr, "qw3: out of memory\n");
            qw3_session_free(session);
            qw3_tokens_free(&tokens);
            qw3_engine_close(engine);
            return 1;
        }
        int n = qw3_session_top_logprobs(session, scores, top_k_logits);
        printf("top%d after prefill:\n", n);
        for (int i = 0; i < n; i++)
        {
            printf("%2d  id=%d  logit=%.7g  text=",
                   i + 1, scores[i].id, scores[i].logit);
            print_token_piece(engine, scores[i].id);
            putchar('\n');
        }
        free(scores);
        qw3_session_free(session);
        qw3_tokens_free(&tokens);
    }
    if (dump_logprobs_path)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --dump-logprobs requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        sample_opts dump_sample = sample;
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int dump_rc = dump_logprobs(engine, &tokens, ctx_size, n_predict,
                                    logprobs_top_k, &dump_sample,
                                    dump_logprobs_path);
        qw3_tokens_free(&tokens);
        if (dump_rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
        printf("dumped logprobs: path=%s steps=%d top_k=%d\n",
               dump_logprobs_path, n_predict,
               logprobs_top_k > 0 ? logprobs_top_k : 20);
    }
    if (save_session_path)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --save-session requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_session *session = NULL;
        char err[256] = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        FILE *fp = fopen(save_session_path, "wb");
        if (!fp ||
            qw3_session_create(&session, engine, ctx_size) != 0 ||
            qw3_session_sync(session, &tokens, err, sizeof(err)) != 0 ||
            qw3_session_save_payload(session, fp, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "qw3: save-session failed: %s\n", err);
            if (fp)
                fclose(fp);
            qw3_session_free(session);
            qw3_tokens_free(&tokens);
            qw3_engine_close(engine);
            return 1;
        }
        uint64_t bytes = qw3_session_payload_bytes(session);
        fclose(fp);
        printf("saved session: path=%s bytes=%llu pos=%d\n",
               save_session_path, (unsigned long long)bytes,
               qw3_session_pos(session));
        qw3_session_free(session);
        qw3_tokens_free(&tokens);
    }
    if (load_session_path)
    {
        qw3_session *session = NULL;
        char err[256] = {0};
        FILE *fp = fopen(load_session_path, "rb");
        uint64_t bytes = 0;
        if (!fp ||
            file_payload_size(fp, &bytes) != 0 ||
            qw3_session_create(&session, engine, ctx_size) != 0 ||
            qw3_session_load_payload(session, fp, bytes, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "qw3: load-session failed: %s\n", err);
            if (fp)
                fclose(fp);
            qw3_session_free(session);
            qw3_engine_close(engine);
            return 1;
        }
        fclose(fp);
        fprintf(stderr, "qw3: loaded session %s (bytes=%llu pos=%d)\n",
                load_session_path, (unsigned long long)bytes,
                qw3_session_pos(session));
        emit_ctx emit = {.engine = engine};
        int gen_rc = generate_from_session(engine, session, n_predict, &emit, &sample);
        qw3_session_free(session);
        if (gen_rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (trace_layers)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --trace-layers requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int trace_rc = qw3_engine_trace_prompt(engine, &tokens, ctx_size, stdout);
        qw3_tokens_free(&tokens);
        if (trace_rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (dump_trace_path)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --dump-trace requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        FILE *fp = fopen(dump_trace_path, "wb");
        if (!fp)
        {
            fprintf(stderr, "qw3: cannot open trace dump %s\n", dump_trace_path);
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int trace_rc = qw3_engine_trace_prompt_json(engine, &tokens, ctx_size, fp);
        fclose(fp);
        qw3_tokens_free(&tokens);
        if (trace_rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
        printf("dumped trace: path=%s\n", dump_trace_path);
    }
    if (metal_rmsnorm_test)
    {
        if (qw3_engine_metal_rmsnorm_test(engine, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_rmsnorm_weight_test >= 0)
    {
        if (qw3_engine_metal_rmsnorm_weight_test(engine, metal_rmsnorm_weight_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_embed_test >= 0)
    {
        if (qw3_engine_metal_embed_test(engine, metal_embed_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_matvec_q8_test >= 0)
    {
        if (qw3_engine_metal_matvec_q8_0_test(engine, metal_matvec_q8_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_proj_test >= 0)
    {
        if (qw3_engine_metal_deltanet_proj_test(engine, metal_deltanet_proj_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_conv_test >= 0)
    {
        if (qw3_engine_metal_deltanet_conv_test(engine, metal_deltanet_conv_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_conv_step_test >= 0)
    {
        if (qw3_engine_metal_deltanet_conv_step_test(engine, metal_deltanet_conv_step_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_l2_test >= 0)
    {
        if (qw3_engine_metal_deltanet_l2norm_test(engine, metal_deltanet_l2_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_gates_test >= 0)
    {
        if (qw3_engine_metal_deltanet_gates_test(engine, metal_deltanet_gates_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_recur_zero_test >= 0)
    {
        if (qw3_engine_metal_deltanet_recur_zero_test(engine, metal_deltanet_recur_zero_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_recur_test >= 0)
    {
        if (qw3_engine_metal_deltanet_recur_test(engine, metal_deltanet_recur_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_recur_step_test >= 0)
    {
        if (qw3_engine_metal_deltanet_recur_step_test(engine, metal_deltanet_recur_step_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_gated_norm_test >= 0)
    {
        if (qw3_engine_metal_deltanet_gated_norm_test(engine, metal_deltanet_gated_norm_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_out_test >= 0)
    {
        if (qw3_engine_metal_deltanet_out_test(engine, metal_deltanet_out_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_branch_step_test >= 0)
    {
        if (qw3_engine_metal_deltanet_branch_step_test(engine, metal_deltanet_branch_step_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_layer_step_test >= 0)
    {
        if (qw3_engine_metal_deltanet_layer_step_test(engine, metal_deltanet_layer_step_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_layer2_test >= 0)
    {
        if (qw3_engine_metal_deltanet_layer2_test(engine, metal_deltanet_layer2_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_layer4_test >= 0)
    {
        if (qw3_engine_metal_deltanet_layer4_test(engine, metal_deltanet_layer4_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_layer8_test >= 0)
    {
        if (qw3_engine_metal_deltanet_layer8_test(engine, metal_deltanet_layer8_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_branch_test >= 0)
    {
        if (qw3_engine_metal_deltanet_branch_test(engine, metal_deltanet_branch_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet_resid_norm_test >= 0)
    {
        if (qw3_engine_metal_deltanet_residual_norm_test(engine, metal_deltanet_resid_norm_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_router_test >= 0)
    {
        if (qw3_engine_metal_moe_router_test(engine, metal_moe_router_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_shared_test >= 0)
    {
        if (qw3_engine_metal_moe_shared_test(engine, metal_moe_shared_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_iq4_down_test >= 0)
    {
        if (qw3_engine_metal_moe_iq4_down_test(engine, metal_moe_iq4_down_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_q6_down_test >= 0)
    {
        if (qw3_engine_metal_moe_q6_down_test(engine, metal_moe_q6_down_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_iq3_test >= 0)
    {
        if (qw3_engine_metal_moe_iq3_test(engine, metal_moe_iq3_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_sparse_top1_test >= 0)
    {
        if (qw3_engine_metal_moe_sparse_top1_test(engine, metal_moe_sparse_top1_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_sparse_top8_test >= 0)
    {
        if (qw3_engine_metal_moe_sparse_top8_test(engine, metal_moe_sparse_top8_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_layer_test >= 0)
    {
        if (qw3_engine_metal_moe_layer_test(engine, metal_moe_layer_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_moe_real_layer_test >= 0)
    {
        if (qw3_engine_metal_moe_real_layer_test(engine, metal_moe_real_layer_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_deltanet3_test >= 0)
    {
        if (qw3_engine_metal_deltanet3_test(engine, metal_deltanet3_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed4_test >= 0)
    {
        if (qw3_engine_metal_mixed4_test(engine, metal_mixed4_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed8_test >= 0)
    {
        if (qw3_engine_metal_mixed8_test(engine, metal_mixed8_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed16_test >= 0)
    {
        if (qw3_engine_metal_mixed16_test(engine, metal_mixed16_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed32_test >= 0)
    {
        if (qw3_engine_metal_mixed32_test(engine, metal_mixed32_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed33_test >= 0)
    {
        if (qw3_engine_metal_mixed33_test(engine, metal_mixed33_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed34_test >= 0)
    {
        if (qw3_engine_metal_mixed34_test(engine, metal_mixed34_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed35_test >= 0)
    {
        if (qw3_engine_metal_mixed35_test(engine, metal_mixed35_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed36_test >= 0)
    {
        if (qw3_engine_metal_mixed36_test(engine, metal_mixed36_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_mixed40_test >= 0)
    {
        if (qw3_engine_metal_mixed40_test(engine, metal_mixed40_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_logits_test >= 0)
    {
        if (qw3_engine_metal_logits_test(engine, metal_logits_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_decode_test)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --metal-decode-test requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int rc = qw3_engine_metal_decode_test(engine, &tokens, ctx_size, stdout);
        qw3_tokens_free(&tokens);
        if (rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_decode_test)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --metal-session-decode-test requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int rc = qw3_engine_metal_session_decode_test(engine, &tokens,
                                                      ctx_size, stdout);
        qw3_tokens_free(&tokens);
        if (rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_test)
    {
        if (qw3_engine_metal_session_test(engine, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_embed_test >= 0)
    {
        if (qw3_engine_metal_session_embed_test(engine, metal_session_embed_test,
                                                ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_rmsnorm_test >= 0)
    {
        if (qw3_engine_metal_session_rmsnorm_test(engine, metal_session_rmsnorm_test,
                                                  ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_qkv_test >= 0)
    {
        if (qw3_engine_metal_session_qkv_test(engine, metal_session_qkv_test,
                                              ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_prefill_q8_batch_test >= 0)
    {
        if (qw3_engine_metal_session_prefill_q8_batch_test(
                engine, metal_session_prefill_q8_batch_test,
                ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_gqa_prefill_batch_test >= 0)
    {
        if (qw3_engine_metal_session_gqa_prefill_batch_test(
                engine, metal_session_gqa_prefill_batch_test,
                ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_z_test >= 0)
    {
        if (qw3_engine_metal_session_z_test(engine, metal_session_z_test,
                                            ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_conv_test >= 0)
    {
        if (qw3_engine_metal_session_conv_test(engine, metal_session_conv_test,
                                               ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_l2norm_test >= 0)
    {
        if (qw3_engine_metal_session_l2norm_test(engine, metal_session_l2norm_test,
                                                 ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_gates_test >= 0)
    {
        if (qw3_engine_metal_session_gates_test(engine, metal_session_gates_test,
                                                ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_recur_zero_test >= 0)
    {
        if (qw3_engine_metal_session_recur_zero_test(engine, metal_session_recur_zero_test,
                                                     ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_recur_step_test >= 0)
    {
        if (qw3_engine_metal_session_recur_step_test(engine, metal_session_recur_step_test,
                                                     ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_gated_rmsnorm_test >= 0)
    {
        if (qw3_engine_metal_session_gated_rmsnorm_test(
                engine, metal_session_gated_rmsnorm_test, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_attn_out_test >= 0)
    {
        if (qw3_engine_metal_session_attn_out_test(
                engine, metal_session_attn_out_test, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_ffn_norm_test >= 0)
    {
        if (qw3_engine_metal_session_ffn_norm_test(
                engine, metal_session_ffn_norm_test, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_layer0_test >= 0)
    {
        if (qw3_engine_metal_session_layer0_test(
                engine, metal_session_layer0_test, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_gqa_project_test >= 0)
    {
        if (qw3_engine_metal_session_gqa_project_test(
                engine, metal_session_gqa_project_test, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_gqa_single_test >= 0)
    {
        if (qw3_engine_metal_session_gqa_single_test(
                engine, metal_session_gqa_single_test, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_gqa_cached2_test >= 0)
    {
        if (qw3_engine_metal_session_gqa_cached2_test(
                engine, metal_session_gqa_cached2_test, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_session_gqa_cached_bench_token >= 0)
    {
        if (qw3_engine_metal_session_gqa_cached_bench(
                engine, metal_session_gqa_cached_bench_token,
                metal_session_gqa_cached_bench_n, ctx_size, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_greedy_test > 0)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --metal-greedy-test requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int rc = qw3_engine_metal_greedy_test(engine, &tokens, ctx_size,
                                              metal_greedy_test, stdout);
        qw3_tokens_free(&tokens);
        if (rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_run > 0)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --metal-run requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int rc = qw3_engine_metal_greedy_run(engine, &tokens, ctx_size,
                                             metal_run, metal_run_quiet,
                                             stdout);
        qw3_tokens_free(&tokens);
        if (rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_gqa_project_test >= 0)
    {
        if (qw3_engine_metal_gqa_project_test(engine, metal_gqa_project_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_gqa_single_test >= 0)
    {
        if (qw3_engine_metal_gqa_single_test(engine, metal_gqa_single_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_gqa_attend2_test >= 0)
    {
        if (qw3_engine_metal_gqa_attend2_test(engine, metal_gqa_attend2_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_gqa_attend4_test >= 0)
    {
        if (qw3_engine_metal_gqa_attend4_test(engine, metal_gqa_attend4_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_gqa_branch4_test >= 0)
    {
        if (qw3_engine_metal_gqa_branch4_test(engine, metal_gqa_branch4_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_gqa_layer4_test >= 0)
    {
        if (qw3_engine_metal_gqa_layer4_test(engine, metal_gqa_layer4_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (metal_gqa_real_layer_test >= 0)
    {
        if (qw3_engine_metal_gqa_real_layer_test(engine, metal_gqa_real_layer_test, stdout) != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (session_roundtrip)
    {
        if (!prompt)
        {
            fprintf(stderr, "qw3: --session-roundtrip requires -p TEXT\n");
            qw3_engine_close(engine);
            return 1;
        }
        qw3_tokens tokens = {0};
        qw3_session *a = NULL;
        qw3_session *b = NULL;
        char err[256] = {0};
        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        FILE *fp = tmpfile();
        if (!fp ||
            qw3_session_create(&a, engine, ctx_size) != 0 ||
            qw3_session_create(&b, engine, ctx_size) != 0 ||
            qw3_session_sync(a, &tokens, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "qw3: roundtrip setup failed: %s\n", err);
            if (fp)
                fclose(fp);
            qw3_session_free(a);
            qw3_session_free(b);
            qw3_tokens_free(&tokens);
            qw3_engine_close(engine);
            return 1;
        }
        uint64_t bytes = qw3_session_payload_bytes(a);
        if (qw3_session_save_payload(a, fp, err, sizeof(err)) != 0 ||
            fseek(fp, 0, SEEK_SET) != 0 ||
            qw3_session_load_payload(b, fp, bytes, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "qw3: roundtrip failed: %s\n", err);
            fclose(fp);
            qw3_session_free(a);
            qw3_session_free(b);
            qw3_tokens_free(&tokens);
            qw3_engine_close(engine);
            return 1;
        }
        qw3_token_score sa[5], sb[5];
        int na = qw3_session_top_logprobs(a, sa, 5);
        int nb = qw3_session_top_logprobs(b, sb, 5);
        float maxdiff = 0.0f;
        int ok = (na == nb);
        for (int i = 0; i < na && i < nb; i++)
        {
            float d = fabsf(sa[i].logit - sb[i].logit);
            if (d > maxdiff)
                maxdiff = d;
            if (sa[i].id != sb[i].id || d != 0.0f)
                ok = 0;
        }
        printf("session roundtrip: bytes=%llu pos=%d top5=%s maxdiff=%.7g\n",
               (unsigned long long)bytes, qw3_session_pos(b),
               ok ? "ok" : "mismatch", maxdiff);
        fclose(fp);
        qw3_session_free(a);
        qw3_session_free(b);
        qw3_tokens_free(&tokens);
        if (!ok)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (prompt && top_k_logits <= 0 && !session_roundtrip && !trace_layers &&
        !dump_trace_path &&
        !metal_rmsnorm_test &&
        metal_rmsnorm_weight_test < 0 &&
        metal_embed_test < 0 &&
        metal_matvec_q8_test < 0 &&
        metal_deltanet_proj_test < 0 &&
        metal_deltanet_conv_test < 0 &&
        metal_deltanet_conv_step_test < 0 &&
        metal_deltanet_l2_test < 0 &&
        metal_deltanet_gates_test < 0 &&
        metal_deltanet_recur_zero_test < 0 &&
        metal_deltanet_recur_test < 0 &&
        metal_deltanet_recur_step_test < 0 &&
        metal_deltanet_gated_norm_test < 0 &&
        metal_deltanet_out_test < 0 &&
        metal_deltanet_branch_step_test < 0 &&
        metal_deltanet_layer_step_test < 0 &&
        metal_deltanet_layer2_test < 0 &&
        metal_deltanet_layer4_test < 0 &&
        metal_deltanet_layer8_test < 0 &&
        metal_deltanet_branch_test < 0 &&
        metal_deltanet_resid_norm_test < 0 &&
        metal_moe_router_test < 0 &&
        metal_moe_shared_test < 0 &&
        metal_moe_iq4_down_test < 0 &&
        metal_moe_q6_down_test < 0 &&
        metal_moe_iq3_test < 0 &&
        metal_moe_sparse_top1_test < 0 &&
        metal_moe_sparse_top8_test < 0 &&
        metal_moe_layer_test < 0 &&
        metal_moe_real_layer_test < 0 &&
        metal_deltanet3_test < 0 &&
        metal_mixed4_test < 0 &&
        metal_mixed8_test < 0 &&
        metal_mixed16_test < 0 &&
        metal_mixed32_test < 0 &&
        metal_mixed33_test < 0 &&
        metal_mixed34_test < 0 &&
        metal_mixed35_test < 0 &&
        metal_mixed36_test < 0 &&
        metal_mixed40_test < 0 &&
        metal_logits_test < 0 &&
        !metal_decode_test &&
        !metal_session_decode_test &&
        !metal_session_test &&
        metal_session_embed_test < 0 &&
        metal_session_rmsnorm_test < 0 &&
        metal_session_qkv_test < 0 &&
        metal_session_prefill_q8_batch_test < 0 &&
        metal_session_gqa_prefill_batch_test < 0 &&
        metal_session_z_test < 0 &&
        metal_session_conv_test < 0 &&
        metal_session_l2norm_test < 0 &&
        metal_session_gates_test < 0 &&
        metal_session_recur_zero_test < 0 &&
        metal_session_gated_rmsnorm_test < 0 &&
        metal_session_attn_out_test < 0 &&
        metal_session_ffn_norm_test < 0 &&
        metal_session_layer0_test < 0 &&
        metal_session_gqa_project_test < 0 &&
        metal_session_gqa_single_test < 0 &&
        metal_session_gqa_cached2_test < 0 &&
        metal_session_gqa_cached_bench_token < 0 &&
        metal_greedy_test <= 0 &&
        metal_run <= 0 &&
        metal_gqa_project_test < 0 &&
        metal_gqa_single_test < 0 &&
        metal_gqa_attend2_test < 0 &&
        metal_gqa_attend4_test < 0 &&
        metal_gqa_branch4_test < 0 &&
        metal_gqa_layer4_test < 0 &&
        metal_gqa_real_layer_test < 0 &&
        !dump_logprobs_path &&
        !save_session_path && !load_session_path &&
        !tokenize && !chat_tokenize && probe_token < 0 &&
        layer_types < 0 && !inspect)
    {
        qw3_tokens tokens = {0};
        qw3_session *session = NULL;
        char err[256] = {0};
        emit_ctx emit = {.engine = engine};

        qw3_encode_chat_prompt(engine, system_prompt, prompt, think_mode, &tokens);
        int gen_rc = -1;
        if (backend == QW3_BACKEND_METAL)
        {
            if (sample.temperature > 0.0f)
            {
                gen_rc = qw3_engine_metal_generate_sample(
                    engine, &tokens, n_predict, ctx_size,
                    sample.temperature, sample.sample_top_k,
                    sample.top_p, sample.min_p, &sample.rng,
                    emit_token, emit_done, &emit);
            }
            else
            {
                gen_rc = qw3_engine_metal_generate_argmax(
                    engine, &tokens, n_predict, ctx_size,
                    emit_token, emit_done, &emit);
            }
        }
        else if (qw3_session_create(&session, engine, ctx_size) == 0 &&
                 qw3_session_sync(session, &tokens, err, sizeof(err)) == 0)
        {
            gen_rc = generate_from_session(engine, session, n_predict, &emit, &sample);
        }
        else
        {
            fprintf(stderr, "qw3: generation prefill failed: %s\n", err);
        }
        qw3_session_free(session);
        qw3_tokens_free(&tokens);
        if (gen_rc != 0)
        {
            qw3_engine_close(engine);
            return 1;
        }
    }
    else
    {
        if (!prompt)
        {
            int chat_rc = interactive_chat(engine, backend, system_prompt,
                                           ctx_size, n_predict, think_mode,
                                           &sample);
            qw3_engine_close(engine);
            free(prompt_owned);
            free(system_owned);
            return chat_rc;
        }
    }
    qw3_engine_close(engine);
    free(prompt_owned);
    free(system_owned);
    return 0;
}
