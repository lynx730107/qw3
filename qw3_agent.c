/* =========================================================================
 * qw3_agent.c - Qwen3 agent client with DSML tools and a lightweight store.
 * ========================================================================= */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "qw3.h"
#include "../linenoise.h"

#define DSML_BAR "\xef\xbd\x9c"
#define DSML_BEGIN "<" DSML_BAR "DSML" DSML_BAR "tool_calls>"
#define DSML_END "</" DSML_BAR "DSML" DSML_BAR "tool_calls>"
#define DSML_INVOKE "<" DSML_BAR "DSML" DSML_BAR "invoke"
#define DSML_INVOKE_CLOSE "</" DSML_BAR "DSML" DSML_BAR "invoke>"
#define DSML_PARAM "<" DSML_BAR "DSML" DSML_BAR "parameter"
#define DSML_PARAM_CLOSE "</" DSML_BAR "DSML" DSML_BAR "parameter>"

#define QWEN_XML_TOOL_CALL_BEGIN "<tool_call>"
#define QWEN_XML_TOOL_CALL_END "</tool_call>"
#define QWEN_XML_FUNCTION_BEGIN "<function="
#define QWEN_XML_FUNCTION_END "</function>"
#define QWEN_XML_PARAMETER_BEGIN "<parameter="
#define QWEN_XML_PARAMETER_END "</parameter>"
#define QWEN_XML_TOOL_RESPONSE_BEGIN "<tool_response>"
#define QWEN_XML_TOOL_RESPONSE_END "</tool_response>"

#define AGENT_COLOR_DIM "\033[2;90m"
#define AGENT_COLOR_RESET "\033[0m"
#define AGENT_COLOR_CODE "\033[33m"

#define QW3_AGENT_N_LAYER 40
#define QW3_AGENT_READ_DEFAULT_LINES 160
#define QW3_AGENT_READ_MAX_LINES 1000
#define QW3_AGENT_SOURCE_READ_MAX_LINES 80
#define QW3_AGENT_SOURCE_READ_LARGE_BYTES 32768
#define QW3_AGENT_SOURCE_READ_TURN_MAX_LINES 80
#define QW3_AGENT_SOURCE_READ_TURN_MAX_CHUNKS 4
#define QW3_AGENT_SOURCE_READ_TRACKED 8
#define QW3_AGENT_MAX_TOOL_ROUNDS 24
#define QW3_AGENT_CODENAV_MAX_BYTES 30000
#define QW3_AGENT_SEMANTIC_MAX_BYTES 24000
#define QW3_AGENT_CODENAV_TIMEOUT_SEC 30.0
#define QW3_AGENT_SEMANTIC_TIMEOUT_SEC 120.0

#define AGENT_STORE_MAGIC "QW3AGKV1"
#define AGENT_STORE_VERSION 1u

typedef struct {
    float temperature;
    int sample_top_k;
    float top_p;
    float min_p;
    float repeat_penalty;
    int repeat_last_n;
    uint64_t rng;
} sample_opts;

typedef struct {
    char *p;
    size_t len;
    size_t cap;
} strbuf;

typedef struct {
    char name[64];
    char *value;
} tool_param;

typedef struct {
    char name[64];
    tool_param params[16];
    int n_params;
} tool_call;

typedef struct {
    tool_call calls[8];
    int n_calls;
} tool_call_list;

typedef struct {
    const char *model_path;
    const char *prompt;
    char *prompt_owned;
    const char *user_system;
    char *user_system_owned;
    char *system_prompt;
    char *store_dir;
    char *conversation;
    char *chdir_path;
    const char *tool_dsml;
    char *tool_dsml_owned;
    const char *tool_native;
    char *tool_native_owned;
    int n_predict;
    int ctx_size;
    int max_tool_rounds;
    bool tools_enabled;
    bool dump_prompt;
    qw3_backend backend;
    qw3_think_mode think_mode;
    sample_opts sample;
} agent_config;

typedef void (*agent_output_fn)(void *ud, const char *s, size_t n);
typedef bool (*agent_interrupt_fn)(void *ud);
typedef void (*agent_progress_fn)(void *ud, const char *phase,
                                  int current, int total, double tps);

typedef struct {
    agent_output_fn write;
    void *ud;
    bool color;
    bool in_think;
    bool in_code;
    char pending[32];
    size_t pending_len;
} agent_renderer;

typedef struct {
    char *path;
    int chunks;
    int lines;
} agent_source_read_budget;

typedef struct {
    qw3_engine *engine;
    qw3_session *session;
    qw3_tokens transcript;
    agent_config cfg;
    char *last_read_path;
    int last_read_next;
    agent_source_read_budget source_reads[QW3_AGENT_SOURCE_READ_TRACKED];
    char *session_id;
    char *session_title;
    time_t session_created;
    time_t session_updated;
    bool session_stripped;
    agent_output_fn output_write;
    void *output_ud;
    agent_output_fn status_write;
    void *status_ud;
    bool output_color;
    agent_interrupt_fn should_interrupt;
    void *interrupt_ud;
    agent_progress_fn progress_update;
    void *progress_ud;
    double progress_start_sec;
} agent_state;

typedef struct {
    qw3_engine *engine;
    qw3_tokens generated;
    strbuf text;
    size_t printed;
    bool printed_any;
    agent_renderer renderer;
} agent_emit_ctx;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t ctx_size;
    uint32_t token_len;
} agent_store_header;

static void sb_init(strbuf *sb) {
    memset(sb, 0, sizeof(*sb));
}

static void sb_free(strbuf *sb) {
    free(sb->p);
    memset(sb, 0, sizeof(*sb));
}

static int sb_reserve(strbuf *sb, size_t need) {
    if (need <= sb->cap) return 0;
    size_t nc = sb->cap ? sb->cap * 2 : 256;
    while (nc < need) nc *= 2;
    char *np = realloc(sb->p, nc);
    if (!np) return -1;
    sb->p = np;
    sb->cap = nc;
    return 0;
}

static int sb_append_n(strbuf *sb, const char *s, size_t n) {
    if (!s || n == 0) return 0;
    if (sb_reserve(sb, sb->len + n + 1) != 0) return -1;
    memcpy(sb->p + sb->len, s, n);
    sb->len += n;
    sb->p[sb->len] = '\0';
    return 0;
}

static int sb_append(strbuf *sb, const char *s) {
    return sb_append_n(sb, s, s ? strlen(s) : 0);
}

static int sb_printf(strbuf *sb, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return -1;
    }
    if (sb_reserve(sb, sb->len + (size_t)n + 1) != 0) {
        va_end(ap2);
        return -1;
    }
    vsnprintf(sb->p + sb->len, sb->cap - sb->len, fmt, ap2);
    va_end(ap2);
    sb->len += (size_t)n;
    return 0;
}

static void agent_write_with_crlf(FILE *fp, const char *s, size_t n) {
    if (!fp || !s || n == 0) return;
    if (!isatty(fileno(fp))) {
        fwrite(s, 1, n, fp);
        return;
    }

    const char *p = s;
    const char *end = s + n;
    while (p < end) {
        if (*p == '\r') {
            if (p + 1 < end && p[1] == '\n') {
                fwrite("\r\n", 1, 2, fp);
                p += 2;
            } else {
                fwrite("\r", 1, 1, fp);
                p++;
            }
        } else if (*p == '\n') {
            fwrite("\r\n", 1, 2, fp);
            p++;
        } else {
            const char *q = p;
            while (q < end && *q != '\r' && *q != '\n') q++;
            fwrite(p, 1, (size_t)(q - p), fp);
            p = q;
        }
    }
}

static void agent_direct_write(void *ud, const char *s, size_t n) {
    FILE *fp = ud ? (FILE *)ud : stdout;
    if (!s || n == 0) return;
    agent_write_with_crlf(fp, s, n);
    fflush(fp);
}

static void agent_output_write(agent_state *a, const char *s, size_t n) {
    if (!s || n == 0) return;
    if (a && a->output_write) {
        a->output_write(a->output_ud, s, n);
    } else {
        agent_direct_write(stdout, s, n);
    }
}

static void agent_status_write(agent_state *a, const char *s, size_t n) {
    if (!s || n == 0) return;
    if (a && a->status_write) {
        a->status_write(a->status_ud, s, n);
    } else {
        agent_direct_write(stderr, s, n);
    }
}

static void agent_statusf(agent_state *a, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n <= 0) {
        va_end(ap2);
        return;
    }
    char stack[512];
    if ((size_t)n < sizeof(stack)) {
        vsnprintf(stack, sizeof(stack), fmt, ap2);
        va_end(ap2);
        agent_status_write(a, stack, (size_t)n);
        return;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        va_end(ap2);
        return;
    }
    vsnprintf(buf, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    agent_status_write(a, buf, (size_t)n);
    free(buf);
}

static bool agent_tool_color_enabled(agent_state *a) {
    const char *tool_color = getenv("QW3_AGENT_TOOL_COLOR");
    if (tool_color && tool_color[0]) {
        return strcmp(tool_color, "0") != 0 &&
               strcmp(tool_color, "false") != 0 &&
               strcmp(tool_color, "off") != 0 &&
               strcmp(tool_color, "no") != 0;
    }
    const char *no_color = getenv("NO_COLOR");
    if (no_color && no_color[0]) return false;
    return a && a->output_color;
}

static void agent_tool_status_begin(agent_state *a) {
    if (agent_tool_color_enabled(a)) {
        agent_status_write(a, AGENT_COLOR_DIM, strlen(AGENT_COLOR_DIM));
    }
}

static void agent_tool_status_end(agent_state *a) {
    if (agent_tool_color_enabled(a)) {
        agent_status_write(a, AGENT_COLOR_RESET, strlen(AGENT_COLOR_RESET));
    }
}

static bool agent_should_interrupt(agent_state *a) {
    return a && a->should_interrupt && a->should_interrupt(a->interrupt_ud);
}

static double agent_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static char *native_tool_response_text(const char *tool_name, const char *value) {
    strbuf sb;
    sb_init(&sb);
    (void)tool_name;
    sb_append(&sb, QWEN_XML_TOOL_RESPONSE_BEGIN "\n");
    sb_append(&sb, value ? value : "");
    if (sb.len && sb.p[sb.len - 1] != '\n') sb_append(&sb, "\n");
    sb_append(&sb, QWEN_XML_TOOL_RESPONSE_END);
    return sb.p;
}

static void sb_append_xml_escaped(strbuf *sb, const char *s) {
    if (!s) return;
    for (const char *p = s; *p; p++) {
        if (*p == '&') sb_append(sb, "&amp;");
        else if (*p == '<') sb_append(sb, "&lt;");
        else if (*p == '>') sb_append(sb, "&gt;");
        else sb_append_n(sb, p, 1);
    }
}

static char *agent_strdup(const char *s);
static char *token_decoded_text(qw3_engine *engine, int token, size_t *out_len);
static void agent_init_transcript(agent_state *a);
static int agent_set_nonblock(int fd);

static char *native_tool_call_text(const tool_call_list *calls)
    __attribute__((unused));
static char *native_tool_call_text(const tool_call_list *calls) {
    strbuf sb;
    sb_init(&sb);
    for (int i = 0; calls && i < calls->n_calls; i++) {
        const tool_call *call = &calls->calls[i];
        sb_append(&sb, QWEN_XML_TOOL_CALL_BEGIN "\n");
        sb_append(&sb, QWEN_XML_FUNCTION_BEGIN);
        sb_append_xml_escaped(&sb, call->name);
        sb_append(&sb, ">\n");
        for (int j = 0; j < call->n_params; j++) {
            const tool_param *param = &call->params[j];
            sb_append(&sb, QWEN_XML_PARAMETER_BEGIN);
            sb_append_xml_escaped(&sb, param->name);
            sb_append(&sb, ">\n");
            sb_append_xml_escaped(&sb, param->value);
            if (param->value && param->value[0] &&
                param->value[strlen(param->value) - 1] != '\n') {
                sb_append(&sb, "\n");
            }
            sb_append(&sb, QWEN_XML_PARAMETER_END "\n");
        }
        sb_append(&sb, QWEN_XML_FUNCTION_END "\n");
        sb_append(&sb, QWEN_XML_TOOL_CALL_END);
        if (i + 1 < calls->n_calls) sb_append(&sb, "\n");
    }
    return sb.p ? sb.p : agent_strdup("");
}

static char *native_tool_declarations(void) {
    strbuf sb;
    sb_init(&sb);
#define TOOL_DECL(NAME, DESC, PARAMS) \
    sb_append(&sb, "{\"type\":\"function\",\"function\":{\"name\":\"" NAME "\",\"description\":\"" DESC "\",\"parameters\":{\"type\":\"object\",\"properties\":{" PARAMS "}}}}\n")
#define TOOL_PARAM(NAME, DESC) \
    "\"" NAME "\":{\"type\":\"string\",\"description\":\"" DESC "\"}"
    TOOL_DECL("read", "Read a precise numbered line range from a text file. Broad or sequential source-file reads are blocked; use get_skeleton, semantic_search, or get_function first.",
              TOOL_PARAM("path", "Path to read") ","
              TOOL_PARAM("start", "Required first 1-based line for source files") ","
              TOOL_PARAM("lines", "Maximum lines to return; source files are capped by context guard"));
    TOOL_DECL("more", "Continue the previous read.",
              TOOL_PARAM("path", "Optional path") ","
              TOOL_PARAM("lines", "Maximum lines to return; default 160"));
    TOOL_DECL("list", "List files below a path.",
              TOOL_PARAM("path", "Directory path") ","
              TOOL_PARAM("depth", "Maximum recursion depth") ","
              TOOL_PARAM("max", "Maximum entries"));
    TOOL_DECL("get_skeleton", "Return a compact codenav semantic outline of a source file. Use this before any line reads on source files.",
              TOOL_PARAM("path", "Source file path"));
    TOOL_DECL("get_function", "Return the exact source for one function or method using codenav. Prefer this over read after get_skeleton or semantic_search.",
              TOOL_PARAM("function_name", "Exact function/method name") ","
              TOOL_PARAM("path", "Optional source file path"));
    TOOL_DECL("semantic_search", "Search code by meaning with colgrep. Use this before broad read/search when you do not know exact names.",
              TOOL_PARAM("query", "Natural-language code search query") ","
              TOOL_PARAM("path", "Optional file or directory path; default current directory") ","
              TOOL_PARAM("results", "Maximum results, 1-50; default 10") ","
              TOOL_PARAM("include", "Optional include glob, for example *.c") ","
              TOOL_PARAM("exclude", "Optional exclude glob") ","
              TOOL_PARAM("code_only", "true/false; default true") ","
              TOOL_PARAM("semantic_only", "true/false; default false") ","
              TOOL_PARAM("content", "true/false; default false"));
    TOOL_DECL("search", "Search text files for a literal pattern.",
              TOOL_PARAM("pattern", "Literal pattern") ","
              TOOL_PARAM("path", "Root path") ","
              TOOL_PARAM("max", "Maximum matches"));
    TOOL_DECL("write", "Create or overwrite a text file.",
              TOOL_PARAM("path", "Path to write") ","
              TOOL_PARAM("content", "File content"));
    TOOL_DECL("edit", "Replace the first exact text occurrence in a file.",
              TOOL_PARAM("path", "Path to edit") ","
              TOOL_PARAM("old", "Old exact text") ","
              TOOL_PARAM("new", "Replacement text"));
    TOOL_DECL("bash", "Run a shell command and return captured output.",
              TOOL_PARAM("cmd", "Shell command"));
#undef TOOL_PARAM
#undef TOOL_DECL
    return sb.p;
}

static char *agent_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static char *read_file_text(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    if (got != (size_t)n) {
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    if (out_len) *out_len = got;
    return buf;
}

static int write_file_text(const char *path, const char *text) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t n = text ? strlen(text) : 0;
    int rc = (fwrite(text ? text : "", 1, n, fp) == n) ? 0 : -1;
    fclose(fp);
    return rc;
}

static int write_file_bytes(const char *path, const char *data, size_t n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    int rc = (fwrite(data ? data : "", 1, n, fp) == n) ? 0 : -1;
    fclose(fp);
    return rc;
}

static int ensure_dir(const char *path) {
    if (!path || !path[0]) return -1;
    if (mkdir(path, 0755) == 0 || errno == EEXIST) return 0;
    return -1;
}

static char *path_join(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t an = strlen(a);
    size_t bn = strlen(b);
    bool slash = an > 0 && a[an - 1] == '/';
    char *out = malloc(an + bn + (slash ? 1 : 2));
    if (!out) return NULL;
    memcpy(out, a, an);
    size_t pos = an;
    if (!slash) out[pos++] = '/';
    memcpy(out + pos, b, bn + 1);
    return out;
}

static char *default_store_dir(void) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    char *root = path_join(home, ".qw3");
    if (!root) return NULL;
    (void)ensure_dir(root);
    char *dir = path_join(root, "agent_store");
    free(root);
    if (dir) (void)ensure_dir(dir);
    return dir;
}

static char *sanitized_name(const char *name) {
    if (!name || !name[0]) name = "default";
    size_t n = strlen(name);
    char *out = malloc(n + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)name[i];
        out[i] = (isalnum(c) || c == '_' || c == '-' || c == '.') ? (char)c : '_';
    }
    out[n] = '\0';
    return out;
}

static char *store_path_ext(agent_state *a, const char *name,
                            const char *ext) {
    char *safe = sanitized_name(name && name[0] ? name : "default");
    if (!safe) return NULL;
    size_t ext_n = ext ? strlen(ext) : 0;
    size_t n = strlen(safe) + ext_n + 1;
    char *file = malloc(n);
    if (!file) {
        free(safe);
        return NULL;
    }
    snprintf(file, n, "%s%s", safe, ext ? ext : "");
    free(safe);
    char *path = path_join(a->cfg.store_dir, file);
    free(file);
    return path;
}

static char *store_path(agent_state *a, const char *name) {
    return store_path_ext(a, name, ".qw3a");
}

static char *store_meta_path(agent_state *a, const char *name) {
    return store_path_ext(a, name, ".meta");
}

static char *store_text_path(agent_state *a, const char *name) {
    return store_path_ext(a, name, ".txt");
}

static uint64_t agent_hash_update(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static char *agent_session_id_from_seed(const char *title,
                                        const char *model_path,
                                        time_t created) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t h = 1469598103934665603ull;
    h = agent_hash_update(h, title ? title : "", strlen(title ? title : ""));
    h = agent_hash_update(h, model_path ? model_path : "",
                          strlen(model_path ? model_path : ""));
    h = agent_hash_update(h, &created, sizeof(created));
    h = agent_hash_update(h, &ts.tv_nsec, sizeof(ts.tv_nsec));
    pid_t pid = getpid();
    h = agent_hash_update(h, &pid, sizeof(pid));
    char *out = malloc(17);
    if (!out) return NULL;
    snprintf(out, 17, "%016llx", (unsigned long long)h);
    return out;
}

static char *agent_title_from_message(const char *msg) {
    const char *p = msg ? msg : "";
    while (*p && isspace((unsigned char)*p)) p++;
    strbuf sb;
    sb_init(&sb);
    while (*p && *p != '\n' && *p != '\r' && sb.len < 72) {
        unsigned char c = (unsigned char)*p++;
        if (c < 32) continue;
        sb_append_n(&sb, (const char *)&c, 1);
    }
    if (!sb.p || !sb.p[0]) {
        sb_free(&sb);
        return agent_strdup("untitled");
    }
    return sb.p;
}

static void agent_set_owned(char **dst, const char *src) {
    char *next = agent_strdup(src ? src : "");
    if (!next) return;
    free(*dst);
    *dst = next;
}

static void agent_clear_session_meta(agent_state *a) {
    free(a->session_id);
    free(a->session_title);
    a->session_id = NULL;
    a->session_title = NULL;
    a->session_created = 0;
    a->session_updated = 0;
    a->session_stripped = false;
}

static void agent_note_user_message(agent_state *a, const char *msg) {
    if (!a->session_created) a->session_created = time(NULL);
    if (!a->session_title) a->session_title = agent_title_from_message(msg);
    a->session_stripped = false;
}

static int agent_ensure_session_id(agent_state *a, const char *name) {
    if (!a->session_created) a->session_created = time(NULL);
    if (!a->session_title) a->session_title = agent_strdup("untitled");
    if (name && name[0]) {
        agent_set_owned(&a->session_id, name);
    } else if (!a->session_id) {
        a->session_id = agent_session_id_from_seed(
            a->session_title, a->cfg.model_path, a->session_created);
    }
    return a->session_id ? 0 : -1;
}

static char *transcript_rendered_text(agent_state *a,
                                      const qw3_tokens *tokens) {
    strbuf out;
    sb_init(&out);
    if (!a || !tokens) return agent_strdup("");
    for (int i = 0; i < tokens->len; i++) {
        size_t n = 0;
        char *t = token_decoded_text(a->engine, tokens->v[i], &n);
        if (t && n) sb_append_n(&out, t, n);
        free(t);
    }
    return out.p ? out.p : agent_strdup("");
}

typedef struct {
    char *id;
    char *title;
    char *model;
    char *backend;
    char *think;
    time_t created;
    time_t updated;
    int ctx;
    int tokens;
    bool tools;
    bool stripped;
} agent_session_meta;

static void session_meta_free(agent_session_meta *m) {
    if (!m) return;
    free(m->id);
    free(m->title);
    free(m->model);
    free(m->backend);
    free(m->think);
    memset(m, 0, sizeof(*m));
}

static bool str_bool(const char *s) {
    return s && (!strcmp(s, "1") || !strcmp(s, "true") ||
                 !strcmp(s, "on") || !strcmp(s, "yes"));
}

static int session_meta_read_path(const char *path, agent_session_meta *m) {
    memset(m, 0, sizeof(*m));
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, fp) >= 0) {
        char *nl = strpbrk(line, "\r\n");
        if (nl) *nl = '\0';
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq++ = '\0';
        if (!strcmp(line, "id")) m->id = agent_strdup(eq);
        else if (!strcmp(line, "title")) m->title = agent_strdup(eq);
        else if (!strcmp(line, "model")) m->model = agent_strdup(eq);
        else if (!strcmp(line, "backend")) m->backend = agent_strdup(eq);
        else if (!strcmp(line, "think")) m->think = agent_strdup(eq);
        else if (!strcmp(line, "created")) m->created = (time_t)strtoll(eq, NULL, 10);
        else if (!strcmp(line, "updated")) m->updated = (time_t)strtoll(eq, NULL, 10);
        else if (!strcmp(line, "ctx")) m->ctx = atoi(eq);
        else if (!strcmp(line, "tokens")) m->tokens = atoi(eq);
        else if (!strcmp(line, "tools")) m->tools = str_bool(eq);
        else if (!strcmp(line, "stripped")) m->stripped = str_bool(eq);
    }
    free(line);
    fclose(fp);
    return m->id ? 0 : -1;
}

static int session_meta_write_path(const char *path,
                                   const agent_session_meta *m) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    fprintf(fp, "version=2\n");
    fprintf(fp, "id=%s\n", m->id ? m->id : "");
    fprintf(fp, "title=%s\n", m->title ? m->title : "untitled");
    fprintf(fp, "created=%lld\n", (long long)m->created);
    fprintf(fp, "updated=%lld\n", (long long)m->updated);
    fprintf(fp, "model=%s\n", m->model ? m->model : "");
    fprintf(fp, "backend=%s\n", m->backend ? m->backend : "");
    fprintf(fp, "ctx=%d\n", m->ctx);
    fprintf(fp, "think=%s\n", m->think ? m->think : "");
    fprintf(fp, "tools=%d\n", m->tools ? 1 : 0);
    fprintf(fp, "tokens=%d\n", m->tokens);
    fprintf(fp, "stripped=%d\n", m->stripped ? 1 : 0);
    int rc = ferror(fp) ? -1 : 0;
    fclose(fp);
    return rc;
}

static int store_read_tokens_path(const char *path, qw3_tokens *out,
                                  uint32_t *ctx_size_out) {
    memset(out, 0, sizeof(*out));
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    agent_store_header h = {0};
    int rc = -1;
    if (fread(&h, 1, sizeof(h), fp) == sizeof(h) &&
        memcmp(h.magic, AGENT_STORE_MAGIC, sizeof(h.magic)) == 0 &&
        h.version == AGENT_STORE_VERSION) {
        rc = 0;
        for (uint32_t i = 0; i < h.token_len; i++) {
            int32_t tok = 0;
            if (fread(&tok, 1, sizeof(tok), fp) != sizeof(tok)) {
                rc = -1;
                break;
            }
            qw3_tokens_push(out, (int)tok);
        }
        if (ctx_size_out) *ctx_size_out = h.ctx_size;
    }
    fclose(fp);
    if (rc != 0) qw3_tokens_free(out);
    return rc;
}

static int store_write_tokens_path(agent_state *a, const char *path) {
    if (!path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        agent_statusf(a, "agent: cannot save %s: %s\n", path, strerror(errno));
        return -1;
    }
    agent_store_header h = {0};
    memcpy(h.magic, AGENT_STORE_MAGIC, sizeof(h.magic));
    h.version = AGENT_STORE_VERSION;
    h.ctx_size = (uint32_t)a->cfg.ctx_size;
    h.token_len = (uint32_t)a->transcript.len;
    int rc = 0;
    if (fwrite(&h, 1, sizeof(h), fp) != sizeof(h)) rc = -1;
    for (int i = 0; rc == 0 && i < a->transcript.len; i++) {
        int32_t tok = (int32_t)a->transcript.v[i];
        if (fwrite(&tok, 1, sizeof(tok), fp) != sizeof(tok)) rc = -1;
    }
    fclose(fp);
    return rc;
}

static int store_write_meta(agent_state *a, const char *id, bool stripped) {
    char *path = store_meta_path(a, id);
    if (!path) return -1;
    agent_session_meta m = {0};
    m.id = (char *)id;
    m.title = a->session_title ? a->session_title : "untitled";
    m.created = a->session_created ? a->session_created : time(NULL);
    m.updated = a->session_updated ? a->session_updated : time(NULL);
    m.model = (char *)a->cfg.model_path;
    m.backend = (char *)qw3_backend_name(a->cfg.backend);
    m.ctx = a->cfg.ctx_size;
    m.think = (char *)qw3_think_mode_name(a->cfg.think_mode);
    m.tools = a->cfg.tools_enabled;
    m.tokens = a->transcript.len;
    m.stripped = stripped;
    int rc = session_meta_write_path(path, &m);
    free(path);
    return rc;
}

static int store_save(agent_state *a, const char *name) {
    if (agent_ensure_session_id(a, name) != 0) return -1;
    a->session_updated = time(NULL);
    const char *id = a->session_id;
    char *text_path = store_text_path(a, id);
    char *token_path = store_path(a, id);
    if (!text_path || !token_path) {
        free(text_path);
        free(token_path);
        return -1;
    }

    char *rendered = transcript_rendered_text(a, &a->transcript);
    int rc = rendered ?
        write_file_bytes(text_path, rendered, strlen(rendered)) : -1;
    free(rendered);
    if (rc == 0 && !a->session_stripped) {
        rc = store_write_tokens_path(a, token_path);
    }
    if (rc == 0 && a->session_stripped) {
        (void)unlink(token_path);
    }
    if (rc == 0) rc = store_write_meta(a, id, a->session_stripped);
    if (rc == 0) {
        agent_statusf(a, "agent: saved %s (%d tokens%s)\n",
                      id, a->transcript.len,
                      a->session_stripped ? ", stripped" : "");
    } else {
        agent_statusf(a, "agent: failed while saving %s\n", id);
    }
    free(text_path);
    free(token_path);
    return rc;
}

static int store_load(agent_state *a, const char *name) {
    char *token_path = store_path(a, name);
    char *meta_path = store_meta_path(a, name);
    char *text_path = store_text_path(a, name);
    if (!token_path || !meta_path || !text_path) {
        free(token_path);
        free(meta_path);
        free(text_path);
        return -1;
    }
    agent_session_meta meta = {0};
    int have_meta = session_meta_read_path(meta_path, &meta) == 0;
    qw3_tokens next = {0};
    int rc = -1;
    uint32_t saved_ctx = 0;
    if (!have_meta || !meta.stripped) {
        rc = store_read_tokens_path(token_path, &next, &saved_ctx);
    }
    if (rc != 0) {
        size_t text_len = 0;
        char *text = read_file_text(text_path, &text_len);
        if (text) {
            qw3_tokenize_rendered_chat(a->engine, text, &next);
            free(text);
            rc = 0;
        }
    }
    if (rc == 0 && next.len > a->cfg.ctx_size) {
        agent_statusf(a, "agent: session %s has %d tokens, ctx=%d\n",
                      name, next.len, a->cfg.ctx_size);
        qw3_tokens_free(&next);
        rc = -1;
    }
    if (rc == 0) {
        qw3_tokens_free(&a->transcript);
        a->transcript = next;
        if (a->session) qw3_session_invalidate(a->session);
        agent_clear_session_meta(a);
        a->session_id = agent_strdup(have_meta && meta.id ? meta.id : name);
        a->session_title = agent_strdup(have_meta && meta.title ? meta.title : name);
        a->session_created = have_meta && meta.created ? meta.created : time(NULL);
        a->session_updated = have_meta && meta.updated ? meta.updated : time(NULL);
        a->session_stripped = have_meta ? meta.stripped : false;
        agent_statusf(a, "agent: switched to %s (%d tokens%s)\n",
                      a->session_id ? a->session_id : name, a->transcript.len,
                      a->session_stripped ? ", stripped/rebuilt" : "");
    }
    (void)saved_ctx;
    session_meta_free(&meta);
    free(token_path);
    free(meta_path);
    free(text_path);
    return rc;
}

typedef struct {
    char *id;
    char *title;
    time_t updated;
    int tokens;
    bool stripped;
} store_list_item;

static int store_list_cmp(const void *a, const void *b) {
    const store_list_item *ia = (const store_list_item *)a;
    const store_list_item *ib = (const store_list_item *)b;
    if (ia->updated > ib->updated) return -1;
    if (ia->updated < ib->updated) return 1;
    return strcmp(ia->id ? ia->id : "", ib->id ? ib->id : "");
}

static int store_list_add(store_list_item **items, int *len, int *cap,
                          const char *id, const char *title,
                          time_t updated, int tokens, bool stripped) {
    if (*len == *cap) {
        int nc = *cap ? *cap * 2 : 16;
        store_list_item *nv = realloc(*items, (size_t)nc * sizeof(*nv));
        if (!nv) return -1;
        *items = nv;
        *cap = nc;
    }
    store_list_item *it = &(*items)[(*len)++];
    memset(it, 0, sizeof(*it));
    it->id = agent_strdup(id);
    it->title = agent_strdup(title && title[0] ? title : "untitled");
    it->updated = updated;
    it->tokens = tokens;
    it->stripped = stripped;
    return it->id ? 0 : -1;
}

static void store_list(agent_state *a) {
    DIR *dir = opendir(a->cfg.store_dir);
    if (!dir) {
        agent_statusf(a, "agent: cannot open store %s\n", a->cfg.store_dir);
        return;
    }
    store_list_item *items = NULL;
    int len = 0;
    int cap = 0;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        size_t n = strlen(de->d_name);
        if (n > 5 && strcmp(de->d_name + n - 5, ".meta") == 0) {
            char id[256];
            snprintf(id, sizeof(id), "%.*s", (int)(n - 5), de->d_name);
            char *mp = path_join(a->cfg.store_dir, de->d_name);
            agent_session_meta meta = {0};
            if (mp && session_meta_read_path(mp, &meta) == 0) {
                store_list_add(&items, &len, &cap,
                               meta.id ? meta.id : id,
                               meta.title ? meta.title : id,
                               meta.updated, meta.tokens, meta.stripped);
            }
            session_meta_free(&meta);
            free(mp);
        } else if (n > 5 && strcmp(de->d_name + n - 5, ".qw3a") == 0) {
            char id[256];
            snprintf(id, sizeof(id), "%.*s", (int)(n - 5), de->d_name);
            char *mp = store_meta_path(a, id);
            struct stat mst;
            if (mp && stat(mp, &mst) != 0) {
                char *tp = path_join(a->cfg.store_dir, de->d_name);
                struct stat tst;
                qw3_tokens toks = {0};
                uint32_t saved_ctx = 0;
                if (tp && stat(tp, &tst) == 0 &&
                    store_read_tokens_path(tp, &toks, &saved_ctx) == 0) {
                    store_list_add(&items, &len, &cap, id, id,
                                   tst.st_mtime, toks.len, false);
                    qw3_tokens_free(&toks);
                }
                free(tp);
            }
            free(mp);
        }
    }
    closedir(dir);
    qsort(items, (size_t)len, sizeof(*items), store_list_cmp);
    agent_statusf(a, "sessions in %s\n", a->cfg.store_dir);
    for (int i = 0; i < len; i++) {
        char tb[32];
        struct tm tmv;
        localtime_r(&items[i].updated, &tmv);
        strftime(tb, sizeof(tb), "%Y-%m-%d %H:%M", &tmv);
        agent_statusf(a, "  %-16s  %s  %7d  %s  %s\n",
                      items[i].id ? items[i].id : "",
                      tb, items[i].tokens,
                      items[i].stripped ? "stripped" : "tokens  ",
                      items[i].title ? items[i].title : "");
        free(items[i].id);
        free(items[i].title);
    }
    if (len == 0) agent_statusf(a, "  (none)\n");
    free(items);
}

static bool store_current_id_matches(agent_state *a, const char *id) {
    return a && a->session_id && id && !strcmp(a->session_id, id);
}

static int store_delete(agent_state *a, const char *name) {
    if (!name || !name[0]) return -1;
    char *token_path = store_path(a, name);
    char *meta_path = store_meta_path(a, name);
    char *text_path = store_text_path(a, name);
    if (!token_path || !meta_path || !text_path) {
        free(token_path);
        free(meta_path);
        free(text_path);
        return -1;
    }
    int removed = 0;
    if (unlink(token_path) == 0) removed++;
    if (unlink(meta_path) == 0) removed++;
    if (unlink(text_path) == 0) removed++;
    free(token_path);
    free(meta_path);
    free(text_path);
    if (removed == 0) {
        agent_statusf(a, "agent: no session files for %s\n", name);
        return -1;
    }
    if (store_current_id_matches(a, name)) {
        agent_init_transcript(a);
        agent_clear_session_meta(a);
    }
    agent_statusf(a, "agent: deleted %s\n", name);
    return 0;
}

static int store_strip(agent_state *a, const char *name) {
    const char *id = (name && name[0]) ? name : a->session_id;
    if (!id || !id[0]) {
        agent_statusf(a, "agent: /strip needs a session id\n");
        return -1;
    }
    char *token_path = store_path(a, id);
    char *meta_path = store_meta_path(a, id);
    char *text_path = store_text_path(a, id);
    if (!token_path || !meta_path || !text_path) {
        free(token_path);
        free(meta_path);
        free(text_path);
        return -1;
    }

    agent_session_meta meta = {0};
    int have_meta = session_meta_read_path(meta_path, &meta) == 0;
    int rc = 0;

    if (store_current_id_matches(a, id)) {
        char *rendered = transcript_rendered_text(a, &a->transcript);
        rc = rendered ? write_file_bytes(text_path, rendered, strlen(rendered)) : -1;
        free(rendered);
        if (!have_meta) {
            meta.id = agent_strdup(id);
            meta.title = agent_strdup(a->session_title ? a->session_title : id);
            meta.created = a->session_created ? a->session_created : time(NULL);
        }
        meta.tokens = a->transcript.len;
    } else {
        struct stat st;
        if (stat(text_path, &st) != 0 || st.st_size == 0) {
            qw3_tokens toks = {0};
            uint32_t saved_ctx = 0;
            if (store_read_tokens_path(token_path, &toks, &saved_ctx) == 0) {
                char *rendered = transcript_rendered_text(a, &toks);
                rc = rendered ?
                    write_file_bytes(text_path, rendered, strlen(rendered)) : -1;
                free(rendered);
                if (!have_meta) {
                    meta.id = agent_strdup(id);
                    meta.title = agent_strdup(id);
                    meta.created = time(NULL);
                    meta.tokens = toks.len;
                }
                qw3_tokens_free(&toks);
            } else {
                rc = -1;
            }
        }
    }

    if (rc == 0) {
        (void)unlink(token_path);
        if (!meta.id) meta.id = agent_strdup(id);
        if (!meta.title) meta.title = agent_strdup(id);
        if (!meta.created) meta.created = time(NULL);
        meta.updated = time(NULL);
        agent_set_owned(&meta.model, a->cfg.model_path);
        agent_set_owned(&meta.backend, qw3_backend_name(a->cfg.backend));
        meta.ctx = a->cfg.ctx_size;
        agent_set_owned(&meta.think, qw3_think_mode_name(a->cfg.think_mode));
        meta.tools = a->cfg.tools_enabled;
        meta.stripped = true;
        rc = session_meta_write_path(meta_path, &meta);
    }
    if (rc == 0 && store_current_id_matches(a, id)) {
        a->session_stripped = true;
        a->session_updated = meta.updated;
    }
    if (rc == 0) {
        agent_statusf(a, "agent: stripped %s\n", id);
    } else {
        agent_statusf(a, "agent: cannot strip %s\n", id);
    }
    session_meta_free(&meta);
    free(token_path);
    free(meta_path);
    free(text_path);
    return rc;
}

static int utf8_read_cp(const char *s, size_t len, size_t *pos, uint32_t *cp) {
    unsigned char c = (unsigned char)s[*pos];
    if (c < 0x80) {
        *cp = c;
        (*pos)++;
        return 1;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        *cp = ((uint32_t)(c & 0x1f) << 6) |
              ((uint32_t)((unsigned char)s[*pos + 1] & 0x3f));
        *pos += 2;
        return 1;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        *cp = ((uint32_t)(c & 0x0f) << 12) |
              ((uint32_t)((unsigned char)s[*pos + 1] & 0x3f) << 6) |
              ((uint32_t)((unsigned char)s[*pos + 2] & 0x3f));
        *pos += 3;
        return 1;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
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

static int gpt2_codepoint_to_byte(uint32_t cp, unsigned char *out) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || cp >= 174) {
        if (cp <= 255) {
            *out = (unsigned char)cp;
            return 1;
        }
    }
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) {
            continue;
        }
        if (cp == 256 + n) {
            *out = (unsigned char)b;
            return 1;
        }
        n++;
    }
    return 0;
}

static char *token_decoded_text(qw3_engine *engine, int token, size_t *out_len) {
    size_t raw_len = 0;
    char *raw = qw3_token_text(engine, token, &raw_len);
    if (!raw) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    char *out = malloc(raw_len + 1);
    if (!out) {
        free(raw);
        if (out_len) *out_len = 0;
        return NULL;
    }
    size_t ip = 0;
    size_t op = 0;
    while (ip < raw_len) {
        uint32_t cp = 0;
        size_t before = ip;
        utf8_read_cp(raw, raw_len, &ip, &cp);
        unsigned char b = 0;
        if (gpt2_codepoint_to_byte(cp, &b)) {
            out[op++] = (char)b;
        } else {
            size_t n = ip - before;
            memcpy(out + op, raw + before, n);
            op += n;
        }
    }
    out[op] = '\0';
    free(raw);
    if (out_len) *out_len = op;
    return out;
}

static char *xml_unescape(const char *s, size_t n) {
    strbuf out;
    sb_init(&out);
    for (size_t i = 0; i < n;) {
        if (i + 5 <= n && !memcmp(s + i, "&amp;", 5)) {
            sb_append_n(&out, "&", 1);
            i += 5;
        } else if (i + 4 <= n && !memcmp(s + i, "&lt;", 4)) {
            sb_append_n(&out, "<", 1);
            i += 4;
        } else if (i + 4 <= n && !memcmp(s + i, "&gt;", 4)) {
            sb_append_n(&out, ">", 1);
            i += 4;
        } else if (i + 6 <= n && !memcmp(s + i, "&quot;", 6)) {
            sb_append_n(&out, "\"", 1);
            i += 6;
        } else if (i + 6 <= n && !memcmp(s + i, "&apos;", 6)) {
            sb_append_n(&out, "'", 1);
            i += 6;
        } else {
            sb_append_n(&out, s + i, 1);
            i++;
        }
    }
    if (!out.p) return agent_strdup("");
    return out.p;
}

static char *read_attr(const char *tag, const char *name) {
    const char *gt = strchr(tag, '>');
    if (!gt) return NULL;
    char pat[96];
    snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p || p > gt) return NULL;
    p += strlen(pat);
    const char *e = strchr(p, '"');
    if (!e || e > gt) return NULL;
    return xml_unescape(p, (size_t)(e - p));
}

static const char *tool_param_value(const tool_call *call, const char *name) {
    for (int i = 0; i < call->n_params; i++) {
        if (!strcmp(call->params[i].name, name)) return call->params[i].value;
    }
    return NULL;
}

static void free_tool_calls(tool_call_list *list) {
    for (int i = 0; i < list->n_calls; i++) {
        for (int j = 0; j < list->calls[i].n_params; j++) {
            free(list->calls[i].params[j].value);
        }
    }
    memset(list, 0, sizeof(*list));
}

static int parse_tool_calls(const char *text, tool_call_list *out) {
    memset(out, 0, sizeof(*out));
    const char *begin = strstr(text, DSML_BEGIN);
    const char *end_block = begin ? strstr(begin, DSML_END) : NULL;
    if (!begin || !end_block) return 0;

    const char *p = begin;
    while (p < end_block && out->n_calls < 8) {
        const char *inv = strstr(p, DSML_INVOKE);
        if (!inv || inv >= end_block) break;
        const char *tag_end = strchr(inv, '>');
        const char *inv_end = tag_end ? strstr(tag_end, DSML_INVOKE_CLOSE) : NULL;
        if (!tag_end || !inv_end || inv_end > end_block) break;

        tool_call *call = &out->calls[out->n_calls];
        char *name = read_attr(inv, "name");
        if (name && name[0]) {
            snprintf(call->name, sizeof(call->name), "%s", name);
            free(name);
        } else {
            free(name);
            p = inv_end + strlen(DSML_INVOKE_CLOSE);
            continue;
        }

        const char *q = tag_end + 1;
        while (q < inv_end && call->n_params < 16) {
            const char *pa = strstr(q, DSML_PARAM);
            if (!pa || pa >= inv_end) break;
            const char *pa_tag_end = strchr(pa, '>');
            const char *pa_end = pa_tag_end ? strstr(pa_tag_end, DSML_PARAM_CLOSE) : NULL;
            if (!pa_tag_end || !pa_end || pa_end > inv_end) break;
            char *pname = read_attr(pa, "name");
            if (pname && pname[0]) {
                tool_param *tp = &call->params[call->n_params++];
                snprintf(tp->name, sizeof(tp->name), "%s", pname);
                tp->value = xml_unescape(pa_tag_end + 1,
                                         (size_t)(pa_end - (pa_tag_end + 1)));
            }
            free(pname);
            q = pa_end + strlen(DSML_PARAM_CLOSE);
        }
        out->n_calls++;
        p = inv_end + strlen(DSML_INVOKE_CLOSE);
    }
    return out->n_calls;
}

static const char *skip_space(const char *p, const char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    return p;
}

static char *parse_jsonish_atom(const char **pp, const char *end) {
    const char *p = skip_space(*pp, end);
    if (p < end && *p == '"') {
        p++;
        strbuf out;
        sb_init(&out);
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                p++;
                if (*p == 'n') sb_append_n(&out, "\n", 1);
                else if (*p == 'r') sb_append_n(&out, "\r", 1);
                else if (*p == 't') sb_append_n(&out, "\t", 1);
                else sb_append_n(&out, p, 1);
                p++;
            } else {
                sb_append_n(&out, p++, 1);
            }
        }
        if (p < end && *p == '"') p++;
        *pp = p;
        return out.p ? out.p : agent_strdup("");
    }
    const char *q = p;
    while (q < end && *q != ',' && *q != '}') q++;
    const char *r = q;
    while (r > p && isspace((unsigned char)r[-1])) r--;
    *pp = q;
    return xml_unescape(p, (size_t)(r - p));
}

static const char *jsonish_object_end(const char *start) {
    if (!start || *start != '{') return NULL;
    int depth = 0;
    bool in_string = false;
    bool esc = false;
    for (const char *p = start; *p; p++) {
        if (in_string) {
            if (esc) esc = false;
            else if (*p == '\\') esc = true;
            else if (*p == '"') in_string = false;
            continue;
        }
        if (*p == '"') {
            in_string = true;
        } else if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            depth--;
            if (depth == 0) return p;
        }
    }
    return NULL;
}

static const char *jsonish_payload_start(const char *text, const char *end) {
    const char *p = skip_space(text ? text : "", end);
    if (p + 3 <= end && !strncmp(p, "```", 3)) {
        p += 3;
        while (p < end && *p != '\n' && *p != '\r') p++;
        while (p < end && (*p == '\n' || *p == '\r')) p++;
        p = skip_space(p, end);
    }
    return p;
}

static void infer_tool_name_from_params(tool_call *call) {
    if (call->name[0]) return;
    if (tool_param_value(call, "cmd") || tool_param_value(call, "command")) {
        snprintf(call->name, sizeof(call->name), "bash");
    } else if (tool_param_value(call, "function_name") ||
               tool_param_value(call, "symbol")) {
        snprintf(call->name, sizeof(call->name), "get_function");
    } else if (tool_param_value(call, "query")) {
        snprintf(call->name, sizeof(call->name), "semantic_search");
    } else if (tool_param_value(call, "pattern")) {
        snprintf(call->name, sizeof(call->name), "search");
    } else if (tool_param_value(call, "old") && tool_param_value(call, "new")) {
        snprintf(call->name, sizeof(call->name), "edit");
    } else if (tool_param_value(call, "content")) {
        snprintf(call->name, sizeof(call->name), "write");
    } else if (tool_param_value(call, "depth") || tool_param_value(call, "max")) {
        snprintf(call->name, sizeof(call->name), "list");
    } else if (tool_param_value(call, "path")) {
        snprintf(call->name, sizeof(call->name), "read");
    }
}

static void parse_jsonish_params_object(const char *start, const char *end,
                                        tool_call *call) {
    const char *p = start;
    while (p < end && call->n_params < 16) {
        p = skip_space(p, end);
        if (p >= end) break;
        char *key = parse_jsonish_atom(&p, end);
        if (!key || !key[0]) {
            free(key);
            break;
        }
        p = skip_space(p, end);
        if (p >= end || *p != ':') {
            free(key);
            break;
        }
        p++;
        p = skip_space(p, end);
        if (p < end && *p == '{') {
            const char *obj_end = jsonish_object_end(p);
            if (!obj_end || obj_end > end) {
                free(key);
                break;
            }
            if (!strcmp(key, "arguments") || !strcmp(key, "parameters") ||
                !strcmp(key, "args")) {
                parse_jsonish_params_object(p + 1, obj_end, call);
            }
            p = obj_end + 1;
            free(key);
        } else {
            char *value = parse_jsonish_atom(&p, end);
            if (!strcmp(key, "name") || !strcmp(key, "tool") ||
                !strcmp(key, "function")) {
                snprintf(call->name, sizeof(call->name), "%s",
                         value ? value : "");
                free(key);
                free(value);
            } else if (strcmp(key, "type")) {
                tool_param *tp = &call->params[call->n_params++];
                snprintf(tp->name, sizeof(tp->name), "%s", key);
                tp->value = value ? value : agent_strdup("");
                free(key);
            } else {
                free(key);
                free(value);
            }
        }
        p = skip_space(p, end);
        if (p < end && *p == ',') p++;
    }
}

static int parse_jsonish_tool_call(const char *text, tool_call_list *out) {
    if (!text) return 0;
    const char *text_end = text + strlen(text);
    const char *start = jsonish_payload_start(text, text_end);
    if (!start || start >= text_end || *start != '{') start = strchr(text, '{');
    const char *end = start ? jsonish_object_end(start) : NULL;
    if (!start || !end || out->n_calls >= 8) return 0;
    tool_call *call = &out->calls[out->n_calls];
    parse_jsonish_params_object(start + 1, end, call);
    infer_tool_name_from_params(call);
    if (!call->name[0]) {
        for (int i = 0; i < call->n_params; i++) free(call->params[i].value);
        memset(call, 0, sizeof(*call));
        return 0;
    }
    out->n_calls++;
    return 1;
}

static int parse_native_tool_calls(const char *text, tool_call_list *out) {
    memset(out, 0, sizeof(*out));
    const char *p = text ? text : "";
    size_t start_len = strlen(QWEN_XML_TOOL_CALL_BEGIN);
    size_t end_len = strlen(QWEN_XML_TOOL_CALL_END);
    while (out->n_calls < 8) {
        const char *start = strstr(p, QWEN_XML_TOOL_CALL_BEGIN);
        if (!start) break;
        const char *close = strstr(start + start_len, QWEN_XML_TOOL_CALL_END);
        if (!close) break;
        const char *body = start + start_len;
        const char *body_end = close;
        body = skip_space(body, body_end);
        if ((size_t)(body_end - body) < strlen(QWEN_XML_FUNCTION_BEGIN) ||
            strncmp(body, QWEN_XML_FUNCTION_BEGIN,
                    strlen(QWEN_XML_FUNCTION_BEGIN)) != 0) {
            p = close + end_len;
            continue;
        }
        body += strlen(QWEN_XML_FUNCTION_BEGIN);
        const char *name_start = body;
        const char *name_end = strchr(body, '>');
        if (!name_end || name_end >= body_end) {
            p = close + end_len;
            continue;
        }
        const char *fn_end = strstr(name_end + 1, QWEN_XML_FUNCTION_END);
        if (!fn_end || fn_end > body_end) fn_end = body_end;
        tool_call *call = &out->calls[out->n_calls];
        snprintf(call->name, sizeof(call->name), "%.*s",
                 (int)(name_end - name_start), name_start);

        const char *q = name_end + 1;
        while (q < fn_end && call->n_params < 16) {
            const char *pa = strstr(q, QWEN_XML_PARAMETER_BEGIN);
            if (!pa || pa >= fn_end) break;
            const char *pa_name = pa + strlen(QWEN_XML_PARAMETER_BEGIN);
            const char *pa_gt = strchr(pa_name, '>');
            if (!pa_gt || pa_gt >= fn_end) break;
            const char *pa_end = strstr(pa_gt + 1, QWEN_XML_PARAMETER_END);
            if (!pa_end || pa_end > fn_end) break;
            tool_param *tp = &call->params[call->n_params++];
            snprintf(tp->name, sizeof(tp->name), "%.*s",
                     (int)(pa_gt - pa_name), pa_name);
            const char *val = pa_gt + 1;
            while (val < pa_end && (*val == '\n' || *val == '\r')) val++;
            const char *val_end = pa_end;
            while (val_end > val && (val_end[-1] == '\n' || val_end[-1] == '\r')) {
                val_end--;
            }
            tp->value = xml_unescape(val, (size_t)(val_end - val));
            q = pa_end + strlen(QWEN_XML_PARAMETER_END);
        }
        out->n_calls++;
        p = close + end_len;
    }
    if (out->n_calls == 0) {
        (void)parse_jsonish_tool_call(text, out);
    }
    return out->n_calls;
}

static int int_param(const tool_call *call, const char *name, int def) {
    const char *v = tool_param_value(call, name);
    return v && v[0] ? atoi(v) : def;
}

static int agent_env_int(const char *name, int def, int min, int max) {
    const char *v = getenv(name);
    if (!v || !v[0]) return def;
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v) return def;
    if (n < min) return min;
    if (n > max) return max;
    return (int)n;
}

static bool bool_param(const tool_call *call, const char *name, bool def) {
    const char *v = tool_param_value(call, name);
    if (!v || !v[0]) return def;
    if (str_bool(v)) return true;
    if (!strcmp(v, "0") || !strcmp(v, "false") || !strcmp(v, "off") ||
        !strcmp(v, "no")) {
        return false;
    }
    return def;
}

static bool agent_has_param_value(const tool_call *call, const char *name) {
    const char *v = tool_param_value(call, name);
    return v && v[0];
}

static bool agent_path_is_source(const char *path) {
    if (!path) return false;
    const char *ext = strrchr(path, '.');
    if (!ext) return false;
    return !strcmp(ext, ".c") || !strcmp(ext, ".h") ||
           !strcmp(ext, ".m") || !strcmp(ext, ".mm") ||
           !strcmp(ext, ".cc") || !strcmp(ext, ".cpp") ||
           !strcmp(ext, ".cxx") || !strcmp(ext, ".hh") ||
           !strcmp(ext, ".hpp") || !strcmp(ext, ".metal") ||
           !strcmp(ext, ".swift") || !strcmp(ext, ".py") ||
           !strcmp(ext, ".js") || !strcmp(ext, ".jsx") ||
           !strcmp(ext, ".ts") || !strcmp(ext, ".tsx") ||
           !strcmp(ext, ".java") || !strcmp(ext, ".rs") ||
           !strcmp(ext, ".go") || !strcmp(ext, ".rb") ||
           !strcmp(ext, ".sh");
}

static char *agent_source_read_guard_message(const char *path,
                                             long long bytes,
                                             int max_lines) {
    strbuf out;
    sb_init(&out);
    sb_printf(&out,
              "context_guard: broad source read blocked for %s (%lld bytes).\n",
              path, bytes);
    sb_append(&out,
              "Use get_skeleton(path) for structure, semantic_search(query,path) "
              "to find relevant symbols, then get_function(function_name,path) "
              "for the exact body.\n");
    sb_printf(&out,
              "Use read only with explicit start and lines<=%d for a precise "
              "line range.\n",
              max_lines);
    return out.p;
}

static void agent_reset_source_read_budget(agent_state *a) {
    if (!a) return;
    for (int i = 0; i < QW3_AGENT_SOURCE_READ_TRACKED; i++) {
        free(a->source_reads[i].path);
        a->source_reads[i].path = NULL;
        a->source_reads[i].chunks = 0;
        a->source_reads[i].lines = 0;
    }
}

static void agent_reset_source_read_budget_for_path(agent_state *a,
                                                    const char *path) {
    if (!a || !path || !path[0]) return;
    for (int i = 0; i < QW3_AGENT_SOURCE_READ_TRACKED; i++) {
        if (a->source_reads[i].path &&
            !strcmp(a->source_reads[i].path, path)) {
            free(a->source_reads[i].path);
            a->source_reads[i].path = NULL;
            a->source_reads[i].chunks = 0;
            a->source_reads[i].lines = 0;
            return;
        }
    }
}

static int agent_source_read_slot(agent_state *a, const char *path,
                                  bool create) {
    if (!a || !path || !path[0]) return -1;
    int empty = -1;
    for (int i = 0; i < QW3_AGENT_SOURCE_READ_TRACKED; i++) {
        if (a->source_reads[i].path &&
            !strcmp(a->source_reads[i].path, path)) {
            return i;
        }
        if (!a->source_reads[i].path && empty < 0) empty = i;
    }
    if (!create || empty < 0) return -1;
    a->source_reads[empty].path = agent_strdup(path);
    if (!a->source_reads[empty].path) return -1;
    a->source_reads[empty].chunks = 0;
    a->source_reads[empty].lines = 0;
    return empty;
}

static char *agent_source_read_budget_message(const char *path,
                                              int used_chunks,
                                              int used_lines,
                                              int max_chunks,
                                              int max_lines) {
    strbuf out;
    sb_init(&out);
    sb_printf(&out,
              "context_guard: source read budget exhausted for %s "
              "(chunks=%d/%d, lines=%d/%d this turn).\n",
              path, used_chunks, max_chunks, used_lines, max_lines);
    sb_append(&out,
              "Stop walking the file in read chunks. Use get_skeleton for the "
              "outline, semantic_search for relevant areas, and get_function "
              "for exact function or method bodies.\n");
    return out.p;
}

static char *agent_apply_source_read_budget(agent_state *a, const char *path,
                                            int *lines, int max_chunks,
                                            int max_lines,
                                            bool *budget_clamped) {
    if (budget_clamped) *budget_clamped = false;
    if (!a || !path || !lines || *lines <= 0) return NULL;
    int idx = agent_source_read_slot(a, path, true);
    if (idx < 0) {
        return agent_strdup("context_guard: too many source files read in this turn; use semantic_search or get_function");
    }
    agent_source_read_budget *b = &a->source_reads[idx];
    int remaining_chunks = max_chunks - b->chunks;
    int remaining_lines = max_lines - b->lines;
    if (remaining_chunks <= 0 || remaining_lines <= 0) {
        return agent_source_read_budget_message(path, b->chunks, b->lines,
                                                max_chunks, max_lines);
    }
    if (*lines > remaining_lines) {
        *lines = remaining_lines;
        if (budget_clamped) *budget_clamped = true;
    }
    return NULL;
}

static void agent_record_source_read(agent_state *a, const char *path,
                                     int emitted) {
    if (!a || !path || emitted <= 0) return;
    int idx = agent_source_read_slot(a, path, true);
    if (idx < 0) return;
    a->source_reads[idx].chunks++;
    a->source_reads[idx].lines += emitted;
}

static char *run_argv_capture(char *const argv[], double timeout_sec,
                              size_t max_bytes) {
    if (!argv || !argv[0]) return agent_strdup("error: empty command");
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        strbuf err;
        sb_init(&err);
        sb_printf(&err, "error: pipe failed: %s", strerror(errno));
        return err.p;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        strbuf err;
        sb_init(&err);
        sb_printf(&err, "error: fork failed: %s", strerror(errno));
        return err.p;
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execvp(argv[0], argv);
        dprintf(STDERR_FILENO, "error: exec %s failed: %s\n",
                argv[0], strerror(errno));
        _exit(127);
    }

    close(pipefd[1]);
    (void)agent_set_nonblock(pipefd[0]);

    strbuf out;
    sb_init(&out);
    int status = 0;
    bool exited = false;
    bool pipe_open = true;
    bool truncated = false;
    bool timed_out = false;
    bool sent_term = false;
    const double t0 = agent_now_sec();

    while (pipe_open || !exited) {
        if (!exited) {
            pid_t wr = waitpid(pid, &status, WNOHANG);
            if (wr == pid) {
                exited = true;
            } else if (wr < 0 && errno != EINTR) {
                exited = true;
            }
        }

        double elapsed = agent_now_sec() - t0;
        if (!exited && timeout_sec > 0.0 && elapsed > timeout_sec) {
            if (!sent_term) {
                kill(pid, SIGTERM);
                sent_term = true;
                timed_out = true;
            } else if (elapsed > timeout_sec + 1.0) {
                kill(pid, SIGKILL);
            }
        }

        if (pipe_open) {
            struct pollfd pfd;
            pfd.fd = pipefd[0];
            pfd.events = POLLIN | POLLHUP;
            pfd.revents = 0;
            int pr = poll(&pfd, 1, 100);
            if (pr > 0 && (pfd.revents & (POLLIN | POLLHUP))) {
                for (;;) {
                    char buf[4096];
                    ssize_t n = read(pipefd[0], buf, sizeof(buf));
                    if (n > 0) {
                        if (out.len < max_bytes) {
                            size_t room = max_bytes - out.len;
                            size_t take = (size_t)n < room ? (size_t)n : room;
                            sb_append_n(&out, buf, take);
                            if (take < (size_t)n) truncated = true;
                        } else {
                            truncated = true;
                        }
                    } else if (n == 0) {
                        pipe_open = false;
                        break;
                    } else {
                        if (errno != EAGAIN && errno != EWOULDBLOCK &&
                            errno != EINTR) {
                            pipe_open = false;
                        }
                        break;
                    }
                }
            } else if (pr < 0 && errno != EINTR) {
                pipe_open = false;
            }
        } else if (!exited) {
            struct timespec ts = {0, 100000000};
            nanosleep(&ts, NULL);
        }
    }
    close(pipefd[0]);
    if (!exited) (void)waitpid(pid, &status, 0);

    if (timed_out) {
        if (out.len && out.p[out.len - 1] != '\n') sb_append(&out, "\n");
        sb_printf(&out, "error: command timed out after %.0fs\n", timeout_sec);
    }
    if (truncated) {
        if (out.len && out.p[out.len - 1] != '\n') sb_append(&out, "\n");
        sb_printf(&out, "... truncated by qw3-agent (%zu byte cap)\n",
                  max_bytes);
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        if (out.len && out.p[out.len - 1] != '\n') sb_append(&out, "\n");
        sb_printf(&out, "exit=%d\n", WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
        if (out.len && out.p[out.len - 1] != '\n') sb_append(&out, "\n");
        sb_printf(&out, "signal=%d\n", WTERMSIG(status));
    }
    return out.p ? out.p : agent_strdup("");
}

static char *tool_read(agent_state *a, const tool_call *call) {
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = tool_param_value(call, "file");
    if (!path || !path[0]) return agent_strdup("error: read requires path");
    int start = int_param(call, "start", 1);
    const int max_lines = agent_env_int("QW3_AGENT_READ_MAX_LINES",
                                        QW3_AGENT_READ_MAX_LINES,
                                        1, 10000);
    const int default_lines = agent_env_int("QW3_AGENT_READ_LINES",
                                            QW3_AGENT_READ_DEFAULT_LINES,
                                            1, max_lines);
    int lines = int_param(call, "lines", default_lines);
    if (start < 1) start = 1;
    if (lines <= 0) lines = default_lines;
    if (lines > max_lines) lines = max_lines;

    const bool is_source = agent_path_is_source(path);
    const bool explicit_start = agent_has_param_value(call, "start");
    const bool explicit_lines = agent_has_param_value(call, "lines");
    const int source_max_lines = agent_env_int(
        "QW3_AGENT_SOURCE_READ_MAX_LINES",
        QW3_AGENT_SOURCE_READ_MAX_LINES, 20, max_lines);
    struct stat st;
    const bool have_stat = stat(path, &st) == 0;
    if (is_source && have_stat &&
        st.st_size >= QW3_AGENT_SOURCE_READ_LARGE_BYTES &&
        (!explicit_start || !explicit_lines)) {
        return agent_source_read_guard_message(path, (long long)st.st_size,
                                               source_max_lines);
    }
    bool source_clamped = false;
    if (is_source && lines > source_max_lines) {
        lines = source_max_lines;
        source_clamped = true;
    }
    const int source_turn_max_lines = agent_env_int(
        "QW3_AGENT_SOURCE_READ_TURN_MAX_LINES",
        QW3_AGENT_SOURCE_READ_TURN_MAX_LINES, source_max_lines,
        max_lines * QW3_AGENT_SOURCE_READ_TRACKED);
    const int source_turn_max_chunks = agent_env_int(
        "QW3_AGENT_SOURCE_READ_TURN_MAX_CHUNKS",
        QW3_AGENT_SOURCE_READ_TURN_MAX_CHUNKS, 1, 32);
    bool budget_clamped = false;
    if (is_source) {
        char *budget_err = agent_apply_source_read_budget(
            a, path, &lines, source_turn_max_chunks, source_turn_max_lines,
            &budget_clamped);
        if (budget_err) return budget_err;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        strbuf err;
        sb_init(&err);
        sb_printf(&err, "error: cannot read %s: %s", path, strerror(errno));
        return err.p;
    }
    strbuf out;
    sb_init(&out);
    sb_printf(&out, "read path=%s start=%d lines=%d\n", path, start, lines);
    if (source_clamped) {
        sb_printf(&out,
                  "context_guard: source read capped at %d lines; prefer "
                  "get_skeleton, semantic_search, or get_function for code "
                  "navigation.\n",
                  source_max_lines);
    }
    if (budget_clamped) {
        sb_printf(&out,
                  "context_guard: source read clipped to remaining per-turn "
                  "budget (%d total lines for this source). Use get_function "
                  "or semantic_search instead of continuing sequential reads.\n",
                  source_turn_max_lines);
    }
    char *line = NULL;
    size_t cap = 0;
    int lno = 1;
    int emitted = 0;
    while (getline(&line, &cap, fp) >= 0) {
        if (lno >= start && emitted < lines) {
            sb_printf(&out, "%5d  %s", lno, line);
            emitted++;
        }
        lno++;
        if (emitted >= lines) break;
    }
    free(line);
    fclose(fp);
    if (is_source) agent_record_source_read(a, path, emitted);
    char *saved_path = agent_strdup(path);
    free(a->last_read_path);
    a->last_read_path = saved_path;
    a->last_read_next = start + emitted;
    if (emitted == 0) sb_append(&out, "(no lines)\n");
    return out.p;
}

static char *tool_more(agent_state *a, const tool_call *call) {
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = a->last_read_path;
    if (!path || !path[0]) return agent_strdup("error: no previous read path");
    tool_call next = *call;
    snprintf(next.name, sizeof(next.name), "read");
    if (!tool_param_value(&next, "path") && next.n_params < 16) {
        tool_param *tp = &next.params[next.n_params++];
        snprintf(tp->name, sizeof(tp->name), "path");
        tp->value = (char *)path;
    }
    char start_buf[32];
    snprintf(start_buf, sizeof(start_buf), "%d", a->last_read_next);
    if (!tool_param_value(&next, "start") && next.n_params < 16) {
        tool_param *tp = &next.params[next.n_params++];
        snprintf(tp->name, sizeof(tp->name), "start");
        tp->value = start_buf;
    }
    return tool_read(a, &next);
}

static void list_dir_rec(strbuf *out, const char *path, int depth,
                         int max_depth, int *count, int max_count) {
    if (*count >= max_count) return;
    DIR *dir = opendir(path);
    if (!dir) {
        sb_printf(out, "error: cannot list %s: %s\n", path, strerror(errno));
        return;
    }
    struct dirent *de;
    while ((de = readdir(dir)) != NULL && *count < max_count) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char *child = path_join(path, de->d_name);
        if (!child) continue;
        struct stat st;
        if (stat(child, &st) == 0) {
            for (int i = 0; i < depth; i++) sb_append(out, "  ");
            sb_printf(out, "%s%s\n", de->d_name, S_ISDIR(st.st_mode) ? "/" : "");
            (*count)++;
            if (S_ISDIR(st.st_mode) && depth + 1 < max_depth) {
                list_dir_rec(out, child, depth + 1, max_depth, count, max_count);
            }
        }
        free(child);
    }
    closedir(dir);
}

static char *tool_list(const tool_call *call) {
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = ".";
    int depth = int_param(call, "depth", 2);
    if (depth < 1) depth = 1;
    if (depth > 6) depth = 6;
    int max = int_param(call, "max", 300);
    if (max < 1 || max > 2000) max = 300;
    strbuf out;
    sb_init(&out);
    sb_printf(&out, "list path=%s depth=%d\n", path, depth);
    int count = 0;
    list_dir_rec(&out, path, 0, depth, &count, max);
    if (count >= max) sb_append(&out, "... truncated\n");
    return out.p;
}

static char *tool_get_skeleton(const tool_call *call) {
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = tool_param_value(call, "file");
    if (!path || !path[0]) {
        return agent_strdup("error: get_skeleton requires path");
    }
    int max_bytes = agent_env_int("QW3_AGENT_CODENAV_MAX_BYTES",
                                  QW3_AGENT_CODENAV_MAX_BYTES,
                                  4096, 200000);
    char *argv[] = {
        "codenav",
        "get_skeleton",
        (char *)path,
        NULL
    };
    strbuf out;
    sb_init(&out);
    sb_printf(&out, "get_skeleton path=%s\n", path);
    char *captured = run_argv_capture(argv, QW3_AGENT_CODENAV_TIMEOUT_SEC,
                                      (size_t)max_bytes);
    sb_append(&out, captured ? captured : "");
    free(captured);
    return out.p;
}

static char *tool_get_function(const tool_call *call) {
    const char *name = tool_param_value(call, "function_name");
    if (!name || !name[0]) name = tool_param_value(call, "name");
    if (!name || !name[0]) name = tool_param_value(call, "symbol");
    if (!name || !name[0]) {
        return agent_strdup("error: get_function requires function_name");
    }
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = tool_param_value(call, "file");
    int max_bytes = agent_env_int("QW3_AGENT_CODENAV_MAX_BYTES",
                                  QW3_AGENT_CODENAV_MAX_BYTES,
                                  4096, 200000);
    char *argv_with_path[] = {
        "codenav",
        "get_function",
        (char *)name,
        (char *)path,
        NULL
    };
    char *argv_no_path[] = {
        "codenav",
        "get_function",
        (char *)name,
        NULL
    };
    strbuf out;
    sb_init(&out);
    if (path && path[0]) {
        sb_printf(&out, "get_function function_name=%s path=%s\n", name, path);
    } else {
        sb_printf(&out, "get_function function_name=%s\n", name);
    }
    char *captured = run_argv_capture(path && path[0] ? argv_with_path
                                                      : argv_no_path,
                                      QW3_AGENT_CODENAV_TIMEOUT_SEC,
                                      (size_t)max_bytes);
    sb_append(&out, captured ? captured : "");
    free(captured);
    return out.p;
}

static char *tool_semantic_search(const tool_call *call) {
    const char *query = tool_param_value(call, "query");
    if (!query || !query[0]) query = tool_param_value(call, "q");
    if (!query || !query[0]) {
        return agent_strdup("error: semantic_search requires query");
    }
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = tool_param_value(call, "paths");
    if (!path || !path[0]) path = ".";
    const char *include = tool_param_value(call, "include");
    const char *exclude = tool_param_value(call, "exclude");
    bool code_only = bool_param(call, "code_only", true);
    bool semantic_only = bool_param(call, "semantic_only", false);
    bool content = bool_param(call, "content", false);
    int results = int_param(call, "results", 10);
    if (results < 1) results = 1;
    if (results > 50) results = 50;
    int max_bytes = agent_env_int("QW3_AGENT_SEMANTIC_MAX_BYTES",
                                  QW3_AGENT_SEMANTIC_MAX_BYTES,
                                  4096, 200000);

    char results_buf[32];
    snprintf(results_buf, sizeof(results_buf), "%d", results);
    char include_arg[512];
    char exclude_arg[512];
    int argc = 0;
    char *argv[20];
    argv[argc++] = "colgrep";
    argv[argc++] = "--results";
    argv[argc++] = results_buf;
    if (code_only) argv[argc++] = "--code-only";
    if (semantic_only) argv[argc++] = "--semantic-only";
    if (content) argv[argc++] = "--content";
    if (include && include[0]) {
        snprintf(include_arg, sizeof(include_arg), "--include=%s", include);
        argv[argc++] = include_arg;
    }
    if (exclude && exclude[0]) {
        snprintf(exclude_arg, sizeof(exclude_arg), "--exclude=%s", exclude);
        argv[argc++] = exclude_arg;
    }
    argv[argc++] = (char *)query;
    argv[argc++] = (char *)path;
    argv[argc] = NULL;

    strbuf out;
    sb_init(&out);
    sb_printf(&out, "semantic_search query=%s path=%s results=%d\n",
              query, path, results);
    char *captured = run_argv_capture(argv, QW3_AGENT_SEMANTIC_TIMEOUT_SEC,
                                      (size_t)max_bytes);
    sb_append(&out, captured ? captured : "");
    free(captured);
    return out.p;
}

static char *tool_write(agent_state *a, const tool_call *call) {
    const char *path = tool_param_value(call, "path");
    const char *content = tool_param_value(call, "content");
    if (!path || !path[0]) return agent_strdup("error: write requires path");
    if (!content) content = "";
    if (write_file_text(path, content) != 0) {
        strbuf err;
        sb_init(&err);
        sb_printf(&err, "error: cannot write %s: %s", path, strerror(errno));
        return err.p;
    }
    agent_reset_source_read_budget_for_path(a, path);
    strbuf out;
    sb_init(&out);
    sb_printf(&out,
              "ok: wrote %s (%zu bytes)\n"
              "verification_hint: read a small explicit line range around the "
              "change if you need to verify formatting; do not walk the file.",
              path, strlen(content));
    return out.p;
}

static char *tool_edit(agent_state *a, const tool_call *call) {
    const char *path = tool_param_value(call, "path");
    const char *old = tool_param_value(call, "old");
    const char *new_text = tool_param_value(call, "new");
    if (!path || !old || !new_text) {
        return agent_strdup("error: edit requires path, old and new");
    }
    size_t n = 0;
    char *file = read_file_text(path, &n);
    if (!file) {
        strbuf err;
        sb_init(&err);
        sb_printf(&err, "error: cannot read %s: %s", path, strerror(errno));
        return err.p;
    }
    char *hit = strstr(file, old);
    if (!hit) {
        free(file);
        return agent_strdup("error: old text not found");
    }
    size_t old_n = strlen(old);
    size_t new_n = strlen(new_text);
    strbuf out_file;
    sb_init(&out_file);
    sb_append_n(&out_file, file, (size_t)(hit - file));
    sb_append_n(&out_file, new_text, new_n);
    sb_append(&out_file, hit + old_n);
    int rc = write_file_text(path, out_file.p ? out_file.p : "");
    sb_free(&out_file);
    free(file);
    if (rc != 0) {
        strbuf err;
        sb_init(&err);
        sb_printf(&err, "error: cannot write %s: %s", path, strerror(errno));
        return err.p;
    }
    agent_reset_source_read_budget_for_path(a, path);
    return agent_strdup(
        "ok: edited first occurrence\n"
        "verification_hint: read a small explicit line range around the "
        "change if you need to verify formatting; do not walk the file.");
}

static bool looks_text_file(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext) return true;
    const char *bin[] = {
        ".o", ".a", ".dylib", ".so", ".png", ".jpg", ".jpeg", ".gif",
        ".pdf", ".gguf", ".bin", ".zip", ".tar", ".gz", NULL
    };
    for (int i = 0; bin[i]; i++) {
        if (!strcmp(ext, bin[i])) return false;
    }
    return true;
}

static bool file_looks_binary(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return true;
    unsigned char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf), fp);
    fclose(fp);
    for (size_t i = 0; i < n; i++) {
        if (buf[i] == 0) return true;
    }
    return false;
}

static void search_file(strbuf *out, const char *path, const char *pattern,
                        int *matches, int max_matches) {
    if (*matches >= max_matches || !looks_text_file(path) ||
        file_looks_binary(path)) return;
    FILE *fp = fopen(path, "rb");
    if (!fp) return;
    char *line = NULL;
    size_t cap = 0;
    int lno = 1;
    while (*matches < max_matches && getline(&line, &cap, fp) >= 0) {
        if (strstr(line, pattern)) {
            sb_printf(out, "%s:%d: %s", path, lno, line);
            (*matches)++;
        }
        lno++;
    }
    free(line);
    fclose(fp);
}

static void search_rec(strbuf *out, const char *path, const char *pattern,
                       int depth, int max_depth, int *matches, int max_matches) {
    if (*matches >= max_matches || depth > max_depth) return;
    struct stat st;
    if (stat(path, &st) != 0) return;
    if (S_ISREG(st.st_mode)) {
        search_file(out, path, pattern, matches, max_matches);
        return;
    }
    if (!S_ISDIR(st.st_mode)) return;
    DIR *dir = opendir(path);
    if (!dir) return;
    struct dirent *de;
    while (*matches < max_matches && (de = readdir(dir)) != NULL) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char *child = path_join(path, de->d_name);
        if (child) {
            search_rec(out, child, pattern, depth + 1, max_depth,
                       matches, max_matches);
            free(child);
        }
    }
    closedir(dir);
}

static char *tool_search(const tool_call *call) {
    const char *pattern = tool_param_value(call, "pattern");
    if (!pattern || !pattern[0]) pattern = tool_param_value(call, "query");
    if (!pattern || !pattern[0]) return agent_strdup("error: search requires pattern");
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = ".";
    int max = int_param(call, "max", 100);
    if (max < 1 || max > 1000) max = 100;
    strbuf out;
    sb_init(&out);
    sb_printf(&out, "search pattern=%s path=%s\n", pattern, path);
    int matches = 0;
    search_rec(&out, path, pattern, 0, 8, &matches, max);
    if (matches == 0) sb_append(&out, "(no matches)\n");
    if (matches >= max) sb_append(&out, "... truncated\n");
    return out.p;
}

static char *tool_bash(const tool_call *call) {
    const char *cmd = tool_param_value(call, "cmd");
    if (!cmd || !cmd[0]) cmd = tool_param_value(call, "command");
    if (!cmd || !cmd[0]) return agent_strdup("error: bash requires cmd");
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        strbuf err;
        sb_init(&err);
        sb_printf(&err, "error: popen failed: %s", strerror(errno));
        return err.p;
    }
    strbuf out;
    sb_init(&out);
    sb_printf(&out, "$ %s\n", cmd);
    char buf[1024];
    size_t total = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        size_t n = strlen(buf);
        if (total + n > 32768) {
            sb_append(&out, "\n... truncated\n");
            break;
        }
        sb_append_n(&out, buf, n);
        total += n;
    }
    int status = pclose(fp);
    if (WIFEXITED(status)) {
        sb_printf(&out, "\nexit=%d\n", WEXITSTATUS(status));
    } else {
        sb_append(&out, "\nexit=unknown\n");
    }
    return out.p;
}

static char *execute_one_tool(agent_state *a, const tool_call *call) {
    if (!a->cfg.tools_enabled) return agent_strdup("error: tools are disabled");
    if (!strcmp(call->name, "read")) return tool_read(a, call);
    if (!strcmp(call->name, "more")) return tool_more(a, call);
    if (!strcmp(call->name, "list")) return tool_list(call);
    if (!strcmp(call->name, "get_skeleton")) return tool_get_skeleton(call);
    if (!strcmp(call->name, "get_function") ||
        !strcmp(call->name, "get_fucttion")) return tool_get_function(call);
    if (!strcmp(call->name, "semantic_search")) return tool_semantic_search(call);
    if (!strcmp(call->name, "write")) return tool_write(a, call);
    if (!strcmp(call->name, "edit")) return tool_edit(a, call);
    if (!strcmp(call->name, "search")) return tool_search(call);
    if (!strcmp(call->name, "bash")) return tool_bash(call);
    strbuf out;
    sb_init(&out);
    sb_printf(&out, "error: unknown tool '%s'", call->name);
    return out.p;
}

static char *execute_tools(agent_state *a, const tool_call_list *calls) {
    strbuf result;
    sb_init(&result);
    for (int i = 0; i < calls->n_calls; i++) {
        const tool_call *call = &calls->calls[i];
        agent_tool_status_begin(a);
        agent_statusf(a, "\n[tool] %s\n", call->name);
        char *out = execute_one_tool(a, call);
        sb_printf(&result, "<tool_result name=\"%s\">\n%s\n</tool_result>\n",
                  call->name, out ? out : "");
        agent_statusf(a, "%s\n", out ? out : "");
        agent_tool_status_end(a);
        free(out);
    }
    return result.p ? result.p : agent_strdup("");
}

static void execute_native_tools_append(agent_state *a,
                                        const tool_call_list *calls) {
    for (int i = 0; i < calls->n_calls; i++) {
        const tool_call *call = &calls->calls[i];
        agent_tool_status_begin(a);
        agent_statusf(a, "\n[tool] %s\n", call->name);
        char *out = execute_one_tool(a, call);
        agent_statusf(a, "%s\n", out ? out : "");
        agent_tool_status_end(a);
        char *response = native_tool_response_text(call->name, out ? out : "");
        qw3_chat_append_message(a->engine, &a->transcript, "user", response);
        free(response);
        free(out);
    }
}

static int run_tool_dsml(agent_state *a, const char *dsml) {
    tool_call_list calls;
    int n = parse_tool_calls(dsml ? dsml : "", &calls);
    if (n <= 0) {
        fprintf(stderr, "agent: no complete DSML tool_calls block found\n");
        free_tool_calls(&calls);
        return 1;
    }
    char *result = execute_tools(a, &calls);
    free_tool_calls(&calls);
    if (result) {
        agent_output_write(a, result, strlen(result));
        if (result[0] && result[strlen(result) - 1] != '\n') {
            agent_output_write(a, "\n", 1);
        }
    }
    free(result);
    return 0;
}

static int run_tool_native(agent_state *a, const char *text) {
    tool_call_list calls;
    int n = parse_native_tool_calls(text ? text : "", &calls);
    if (n <= 0) {
        fprintf(stderr, "agent: no complete Qwen native tool_call block found\n");
        free_tool_calls(&calls);
        return 1;
    }
    strbuf result;
    sb_init(&result);
    for (int i = 0; i < calls.n_calls; i++) {
        const tool_call *call = &calls.calls[i];
        agent_tool_status_begin(a);
        agent_statusf(a, "\n[tool] %s\n", call->name);
        char *out = execute_one_tool(a, call);
        agent_statusf(a, "%s\n", out ? out : "");
        agent_tool_status_end(a);
        char *response = native_tool_response_text(call->name, out ? out : "");
        sb_append(&result, response ? response : "");
        if (i + 1 < calls.n_calls) sb_append(&result, "\n");
        free(response);
        free(out);
    }
    free_tool_calls(&calls);
    if (result.p) {
        agent_output_write(a, result.p, result.len);
        if (result.p[0] && result.p[result.len - 1] != '\n') {
            agent_output_write(a, "\n", 1);
        }
    }
    sb_free(&result);
    return 0;
}

static bool agent_prefix_match(const char *buf, size_t n, const char *lit) {
    size_t ln = strlen(lit);
    return n <= ln && memcmp(buf, lit, n) == 0;
}

static void agent_renderer_raw(agent_renderer *r, const char *s, size_t n) {
    if (!r || !s || n == 0) return;
    if (r->write) r->write(r->ud, s, n);
}

static void agent_renderer_apply(agent_renderer *r) {
    if (!r || !r->color) return;
    if (r->in_think) {
        agent_renderer_raw(r, AGENT_COLOR_DIM, strlen(AGENT_COLOR_DIM));
    } else if (r->in_code) {
        agent_renderer_raw(r, AGENT_COLOR_CODE, strlen(AGENT_COLOR_CODE));
    } else {
        agent_renderer_raw(r, AGENT_COLOR_RESET, strlen(AGENT_COLOR_RESET));
    }
}

static void agent_renderer_init(agent_renderer *r, agent_output_fn write,
                                void *ud, bool color, bool initial_think) {
    memset(r, 0, sizeof(*r));
    r->write = write ? write : agent_direct_write;
    r->ud = write ? ud : stdout;
    r->color = color;
    r->in_think = initial_think;
    if (initial_think) agent_renderer_apply(r);
}

static void agent_renderer_emit_byte(agent_renderer *r, char c) {
    if (c == '`') {
        r->in_code = !r->in_code;
        agent_renderer_apply(r);
        return;
    }
    agent_renderer_raw(r, &c, 1);
}

static void agent_renderer_flush_pending(agent_renderer *r) {
    if (!r || r->pending_len == 0) return;
    for (size_t i = 0; i < r->pending_len; i++) {
        agent_renderer_emit_byte(r, r->pending[i]);
    }
    r->pending_len = 0;
}

static void agent_renderer_feed(agent_renderer *r, const char *s, size_t n) {
    if (!r || !s || n == 0) return;
    const char *think_begin = "<think>";
    const char *think_end = "</think>";
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (r->pending_len > 0 || c == '<') {
            if (r->pending_len + 1 < sizeof(r->pending)) {
                r->pending[r->pending_len++] = c;
                r->pending[r->pending_len] = '\0';
            } else {
                agent_renderer_flush_pending(r);
                agent_renderer_emit_byte(r, c);
                continue;
            }

            if (!strcmp(r->pending, think_begin)) {
                r->pending_len = 0;
                r->in_think = true;
                agent_renderer_apply(r);
            } else if (!strcmp(r->pending, think_end)) {
                r->pending_len = 0;
                r->in_think = false;
                agent_renderer_apply(r);
            } else if (agent_prefix_match(r->pending, r->pending_len, think_begin) ||
                       agent_prefix_match(r->pending, r->pending_len, think_end)) {
                continue;
            } else {
                agent_renderer_flush_pending(r);
            }
        } else {
            agent_renderer_emit_byte(r, c);
        }
    }
}

static void agent_renderer_finish(agent_renderer *r) {
    if (!r) return;
    agent_renderer_flush_pending(r);
    if (r->color && (r->in_think || r->in_code)) {
        r->in_think = false;
        r->in_code = false;
        agent_renderer_apply(r);
    }
}

static bool range_is_space(const char *p, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (!isspace((unsigned char)p[i])) return false;
    }
    return true;
}

static bool text_has_jsonish_tool_call(const char *text) {
    tool_call_list calls;
    memset(&calls, 0, sizeof(calls));
    int n = parse_jsonish_tool_call(text ? text : "", &calls);
    free_tool_calls(&calls);
    return n > 0;
}

static bool partial_marker_after_space(const char *text, size_t len,
                                       const char *marker) {
    size_t off = 0;
    while (off < len && isspace((unsigned char)text[off])) off++;
    size_t remain = len - off;
    if (remain == 0) return true;
    size_t marker_len = strlen(marker);
    if (remain >= marker_len) return false;
    return strncmp(text + off, marker, remain) == 0;
}

static bool should_hold_tool_candidate(const char *text, size_t len,
                                       bool final) {
    if (!text || len == 0 || final) return false;
    if (partial_marker_after_space(text, len, DSML_BEGIN) ||
        partial_marker_after_space(text, len, QWEN_XML_TOOL_CALL_BEGIN)) {
        return true;
    }

    const char *end = text + len;
    const char *p = jsonish_payload_start(text, end);
    if (p >= end) return true;
    if (*p != '{') return false;
    if (text_has_jsonish_tool_call(p)) return true;
    return jsonish_object_end(p) == NULL;
}

static bool should_suppress_jsonish_tool_call(const char *text, size_t len) {
    if (!text || len == 0) return false;
    const char *end = text + len;
    const char *p = jsonish_payload_start(text, end);
    if (p >= end || *p != '{') return false;
    return text_has_jsonish_tool_call(p);
}

static bool generated_has_complete_tool_call(const char *text) {
    if (!text) return false;
    return strstr(text, DSML_END) ||
           strstr(text, QWEN_XML_TOOL_CALL_END) ||
           text_has_jsonish_tool_call(text);
}

static size_t marker_suffix_hold_len(const char *text, size_t len,
                                     const char *marker) {
    if (!text || !marker) return 0;
    size_t marker_len = strlen(marker);
    size_t max = len < marker_len ? len : marker_len - 1;
    for (size_t n = max; n > 0; n--) {
        if (!memcmp(text + len - n, marker, n)) return n;
    }
    return 0;
}

static size_t tool_marker_suffix_hold_len(const char *text, size_t len) {
    size_t a = marker_suffix_hold_len(text, len, DSML_BEGIN);
    size_t b = marker_suffix_hold_len(text, len, QWEN_XML_TOOL_CALL_BEGIN);
    return a > b ? a : b;
}

static void agent_flush_visible(agent_emit_ctx *ctx, bool final) {
    const char *dsml = ctx->text.p ? strstr(ctx->text.p, DSML_BEGIN) : NULL;
    const char *native = ctx->text.p ? strstr(ctx->text.p, QWEN_XML_TOOL_CALL_BEGIN) : NULL;
    const char *hidden = NULL;
    if (dsml && native) hidden = dsml < native ? dsml : native;
    else hidden = dsml ? dsml : native;
    size_t limit = 0;
    if (hidden) {
        size_t hidden_off = (size_t)(hidden - ctx->text.p);
        limit = range_is_space(ctx->text.p, hidden_off) ? 0 : hidden_off;
    } else if (should_suppress_jsonish_tool_call(ctx->text.p, ctx->text.len)) {
        limit = 0;
    } else if (should_hold_tool_candidate(ctx->text.p, ctx->text.len, final)) {
        limit = 0;
    } else if (final) {
        limit = ctx->text.len;
    } else {
        size_t hold = tool_marker_suffix_hold_len(ctx->text.p, ctx->text.len);
        limit = ctx->text.len > hold ? ctx->text.len - hold : 0;
    }
    if (limit > ctx->printed) {
        agent_renderer_feed(&ctx->renderer,
                            ctx->text.p + ctx->printed,
                            limit - ctx->printed);
        ctx->printed_any = true;
        ctx->printed = limit;
    }
}

static void agent_emit_token(void *ud, int token) {
    agent_emit_ctx *ctx = (agent_emit_ctx *)ud;
    qw3_tokens_push(&ctx->generated, token);
    size_t len = 0;
    char *text = token_decoded_text(ctx->engine, token, &len);
    if (text && len) {
        sb_append_n(&ctx->text, text, len);
        agent_flush_visible(ctx, false);
    }
    free(text);
}

static void agent_emit_done(void *ud) {
    agent_emit_ctx *ctx = (agent_emit_ctx *)ud;
    agent_flush_visible(ctx, true);
    agent_renderer_finish(&ctx->renderer);
    if (ctx->printed_any) {
        agent_renderer_raw(&ctx->renderer, "\n", 1);
    }
}

static void append_generated_assistant(agent_state *a,
                                       const qw3_tokens *generated) {
    for (int i = 0; i < generated->len; i++) {
        qw3_tokens_push(&a->transcript, generated->v[i]);
    }
}

static void agent_prefill_progress(void *ud, const char *event,
                                   int current, int total) {
    agent_state *a = (agent_state *)ud;
    if (!a || !event || strcmp(event, "prefill_chunk")) return;
    double elapsed = agent_now_sec() - a->progress_start_sec;
    double tps = elapsed > 0.001 ? (double)current / elapsed : 0.0;
    if (a->progress_update) {
        a->progress_update(a->progress_ud, "prefill", current, total, tps);
    }
}

static int generate_once(agent_state *a, char **assistant_text) {
    *assistant_text = NULL;
    qw3_chat_append_assistant_prefix(a->engine, &a->transcript,
                                     a->cfg.think_mode);

    agent_emit_ctx emit;
    memset(&emit, 0, sizeof(emit));
    emit.engine = a->engine;
    agent_renderer_init(&emit.renderer,
                        a->output_write ? a->output_write : agent_direct_write,
                        a->output_write ? a->output_ud : stdout,
                        a->output_color && isatty(STDOUT_FILENO),
                        qw3_think_mode_enabled(a->cfg.think_mode));

    int rc = -1;
    char err[256] = {0};
    int session_pos = qw3_session_pos(a->session);
    int common = qw3_session_common_prefix(a->session, &a->transcript);
    int cached = (common == session_pos && a->transcript.len >= common) ?
                 common : 0;
    const double t_prefill0 = agent_now_sec();
    a->progress_start_sec = t_prefill0;
    qw3_session_set_progress(a->session, agent_prefill_progress, a);
    if (qw3_session_sync(a->session, &a->transcript, err, sizeof(err)) != 0) {
        qw3_session_set_progress(a->session, NULL, NULL);
        if (a->progress_update) {
            a->progress_update(a->progress_ud, "prefill_done", 0, 0, 0.0);
        }
        agent_statusf(a, "agent: prefill failed: %s\n", err);
        rc = -1;
    } else {
        qw3_session_set_progress(a->session, NULL, NULL);
        const double t_prefill1 = agent_now_sec();
        const int prefill_tokens_done = a->transcript.len - cached;
        if (a->progress_update) {
            double prefill_s_done = t_prefill1 - t_prefill0;
            double tps_done = prefill_s_done > 0.0 ?
                (double)prefill_tokens_done / prefill_s_done : 0.0;
            a->progress_update(a->progress_ud, "prefill_done",
                               prefill_tokens_done, prefill_tokens_done,
                               tps_done);
        }
        rc = 0;
        const int eos = qw3_token_eos(a->engine);
        int n_generated = 0;
        const double t_gen0 = agent_now_sec();
        for (int i = 0; i < a->cfg.n_predict; i++) {
            if (agent_should_interrupt(a)) break;
            const int repeat_last_n = a->cfg.sample.repeat_last_n;
            const int generated_len = emit.generated.len;
            int repeat_len = generated_len;
            const int *repeat_tokens = emit.generated.v;
            if (repeat_last_n <= 0 || a->cfg.sample.repeat_penalty <= 1.0f) {
                repeat_len = 0;
                repeat_tokens = NULL;
            } else if (repeat_len > repeat_last_n) {
                repeat_tokens = emit.generated.v + (repeat_len - repeat_last_n);
                repeat_len = repeat_last_n;
            }
            int token = qw3_session_sample_repetition(
                a->session, a->cfg.sample.temperature,
                a->cfg.sample.sample_top_k, a->cfg.sample.top_p,
                a->cfg.sample.min_p, &a->cfg.sample.rng,
                repeat_tokens, repeat_len, a->cfg.sample.repeat_penalty);
            if (token < 0) {
                rc = -1;
                break;
            }
            if (token == eos) break;
            agent_emit_token(&emit, token);
            n_generated++;
            if (qw3_session_eval(a->session, token, err, sizeof(err)) != 0) {
                agent_statusf(a, "agent: decode failed: %s\n", err);
                rc = -1;
                break;
            }
            if (generated_has_complete_tool_call(emit.text.p)) {
                break;
            }
        }
        const double t_gen1 = agent_now_sec();
        agent_emit_done(&emit);

        const int prefill_tokens = a->transcript.len - cached;
        const double prefill_s = t_prefill1 - t_prefill0;
        const double gen_s = t_gen1 - t_gen0;
        agent_statusf(a,
                      "qw3-agent: %s session timing: cached=%d prompt=%d "
                      "prefill=%d tokens %.1f ms (%.2f tok/s) | "
                      "generation=%d tokens %.1f ms (%.2f tok/s)\n",
                      qw3_backend_name(a->cfg.backend), cached,
                      a->transcript.len, prefill_tokens, prefill_s * 1000.0,
                      prefill_s > 0.0 ? (double)prefill_tokens / prefill_s : 0.0,
                      n_generated, gen_s * 1000.0,
                      gen_s > 0.0 ? (double)n_generated / gen_s : 0.0);
    }

    if (rc == 0) {
        append_generated_assistant(a, &emit.generated);
        *assistant_text = emit.text.p ? emit.text.p : agent_strdup("");
        emit.text.p = NULL;
    }
    qw3_tokens_free(&emit.generated);
    sb_free(&emit.text);
    return rc;
}

static int run_agent_turn(agent_state *a, const char *user_message) {
    agent_reset_source_read_budget(a);
    agent_note_user_message(a, user_message);
    qw3_chat_append_message(a->engine, &a->transcript, "user", user_message);
    if (a->transcript.len >= a->cfg.ctx_size) {
        agent_statusf(a, "agent: context full (%d/%d tokens)\n",
                      a->transcript.len, a->cfg.ctx_size);
        return -1;
    }

    for (int round = 0; round < a->cfg.max_tool_rounds; round++) {
        char *assistant = NULL;
        if (generate_once(a, &assistant) != 0) {
            free(assistant);
            return -1;
        }
        if (agent_should_interrupt(a)) {
            free(assistant);
            return 0;
        }
        tool_call_list native_calls;
        int n_native = parse_native_tool_calls(assistant ? assistant : "",
                                               &native_calls);
        if (n_native > 0) {
            execute_native_tools_append(a, &native_calls);
            free_tool_calls(&native_calls);
            free(assistant);
            continue;
        }
        free_tool_calls(&native_calls);

        tool_call_list dsml_calls;
        int n_dsml = parse_tool_calls(assistant ? assistant : "", &dsml_calls);
        if (n_dsml <= 0) {
            free_tool_calls(&dsml_calls);
            free(assistant);
            return 0;
        }
        char *tool_result = execute_tools(a, &dsml_calls);
        free_tool_calls(&dsml_calls);
        char *response = native_tool_response_text("dsml", tool_result);
        qw3_chat_append_message(a->engine, &a->transcript, "user", response);
        free(response);
        free(tool_result);
        free(assistant);
    }
    agent_statusf(a, "agent: max tool rounds reached (%d)\n",
                  a->cfg.max_tool_rounds);
    return 0;
}

static char *build_system_prompt(const char *user_system, bool tools_enabled) {
    strbuf sb;
    sb_init(&sb);
    if (tools_enabled) {
        sb_append(&sb,
            "# Tools\n\n"
            "You have access to the following functions:\n\n"
            "<tools>\n");
        char *tools = native_tool_declarations();
        sb_append(&sb, tools ? tools : "");
        free(tools);
        sb_append(&sb,
            "</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format "
            "with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner "
            "<function=...></function> block must be nested within "
            "<tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in "
            "natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question "
            "like normal with your current knowledge and do not tell the user "
            "about function calls\n"
            "</IMPORTANT>\n\n"
            "Context discipline:\n"
            "- Do not use read(path) to inspect source files broadly; broad "
            "source reads are rejected by the context guard.\n"
            "- Do not walk source files with repeated read chunks; each turn "
            "has a small per-source read budget.\n"
            "- Use get_skeleton before any source line reads when inspecting "
            "file structure.\n"
            "- Prefer get_function when you need one function or method body.\n"
            "- Prefer semantic_search when you know the intent but not the "
            "exact symbol name.\n"
            "- Use read only for small non-source files or precise source line "
            "ranges with explicit start and lines.\n\n");
    }
    sb_append(&sb,
        "You are qw3-agent, a local coding assistant. Work in the current "
        "project, be concise, and be careful with file changes.\n");
    if (user_system && user_system[0]) {
        sb_append(&sb, "\n");
        sb_append(&sb, user_system);
        sb_append(&sb, "\n");
    }
    return sb.p;
}

static void print_help(void) {
    fprintf(stderr,
        "qw3-agent - Qwen3 local agent\n\n"
        "Usage:\n"
        "  qw3-agent -m MODEL [options]\n\n"
        "Options:\n"
        "  -m PATH              Model GGUF path\n"
        "  -p TEXT              Run one prompt and exit\n"
        "  --prompt-file PATH   Read prompt from a file\n"
        "  -sys TEXT            Extra system prompt\n"
        "  --system-file PATH   Read extra system prompt from a file\n"
        "  -n N                 Max tokens per assistant turn (default: 768)\n"
        "  --ctx N              Context size (default: 32768)\n"
        "  --ngl N              Metal layers to keep on GPU, 0..40 (default: 40)\n"
        "  -ctk TYPE -ctv TYPE  Metal KV cache type: f32, f16, or q8_0\n"
        "  --kv-f16             Use f16 Metal GQA KV cache (recommended for large ctx)\n"
        "  --kv-f32             Use f32 Metal GQA KV cache\n"
        "  --kv-q8              Use q8_0 Metal GQA KV cache (experimental)\n"
        "  --temp N             Temperature (default: 0.6)\n"
        "  --sample-top-k N     Sampling top-k (default: 20)\n"
        "  --top-p N            Sampling top-p (default: 0.95)\n"
        "  --min-p N            Sampling min-p (default: 0)\n"
        "  --repeat-penalty N   Repetition penalty (default: 1 disables)\n"
        "  --repeat-last-n N    Generated tokens to penalize (default: 256)\n"
        "  --seed N             Sampling seed\n"
        "  --dump-prompt        Print the rendered prompt and exit\n"
        "  --cpu                Use CPU backend\n"
        "  --metal              Use Metal backend\n"
        "  --nothink            Disable thinking mode\n"
        "  --store-dir PATH     Conversation store directory\n"
        "  --conversation NAME  Load/save a named conversation\n"
        "  --chdir PATH         Change working directory before loading/running\n"
        "  --max-tool-rounds N  Maximum tool/assistant cycles (default: 24)\n"
        "  --no-tools           Disable tool execution\n"
        "  --tool-dsml TEXT     Execute a literal DSML tool_calls block and exit\n"
        "  --tool-dsml-file PATH\n"
        "                       Execute DSML tool_calls read from a file and exit\n"
        "  --tool-native TEXT   Execute literal Qwen <tool_call> XML and exit\n"
        "  --tool-native-file PATH\n"
        "                       Execute Qwen tool_call text read from a file and exit\n"
        "  --help               Show help\n\n"
        "Interactive commands:\n"
        "  /help, /quit, /new, /ctx, /save [name], /list, /switch id\n"
        "  /del id, /strip [id], /load name, /sessions\n"
        "  /read PATH, /think, /nothink, /tools on|off\n");
}

static int parse_args(agent_config *cfg, int argc, char **argv) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->model_path = "./qw3.gguf";
    cfg->n_predict = 768;
    cfg->ctx_size = 32768;
    cfg->max_tool_rounds = QW3_AGENT_MAX_TOOL_ROUNDS;
    cfg->tools_enabled = true;
    cfg->think_mode = QW3_THINK_ON;
    cfg->sample.temperature = 0.6f;
    cfg->sample.sample_top_k = 20;
    cfg->sample.top_p = 0.95f;
    cfg->sample.min_p = 0.0f;
    cfg->sample.repeat_penalty = 1.0f;
    cfg->sample.repeat_last_n = 256;
    cfg->sample.rng = 0x123456789abcdef0ull;
#ifdef QW3_NO_METAL
    cfg->backend = QW3_BACKEND_CPU;
#else
    cfg->backend = QW3_BACKEND_METAL;
#endif
    const char *cache_type_k = NULL;
    const char *cache_type_v = NULL;
    const char *cache_type_alias = NULL;
    int ngl = -1;
    int ngl_set = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_help();
            exit(0);
        } else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            cfg->prompt = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            const char *path = argv[++i];
            cfg->prompt_owned = read_file_text(path, NULL);
            if (!cfg->prompt_owned) {
                fprintf(stderr, "agent: cannot read prompt file %s\n", path);
                return -1;
            }
            cfg->prompt = cfg->prompt_owned;
        } else if ((!strcmp(argv[i], "-sys") || !strcmp(argv[i], "--system")) &&
                   i + 1 < argc) {
            cfg->user_system = argv[++i];
        } else if (!strcmp(argv[i], "--system-file") && i + 1 < argc) {
            const char *path = argv[++i];
            cfg->user_system_owned = read_file_text(path, NULL);
            if (!cfg->user_system_owned) {
                fprintf(stderr, "agent: cannot read system file %s\n", path);
                return -1;
            }
            cfg->user_system = cfg->user_system_owned;
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            cfg->n_predict = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            cfg->ctx_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ngl") && i + 1 < argc) {
            ngl = atoi(argv[++i]);
            ngl_set = 1;
            cfg->backend = QW3_BACKEND_METAL;
        } else if ((!strcmp(argv[i], "-ctk") || !strcmp(argv[i], "--ctk")) &&
                   i + 1 < argc) {
            cache_type_k = argv[++i];
            cfg->backend = QW3_BACKEND_METAL;
        } else if ((!strcmp(argv[i], "-ctv") || !strcmp(argv[i], "--ctv")) &&
                   i + 1 < argc) {
            cache_type_v = argv[++i];
            cfg->backend = QW3_BACKEND_METAL;
        } else if (!strcmp(argv[i], "--kv-f16")) {
            cache_type_alias = "f16";
            cfg->backend = QW3_BACKEND_METAL;
        } else if (!strcmp(argv[i], "--kv-f32")) {
            cache_type_alias = "f32";
            cfg->backend = QW3_BACKEND_METAL;
        } else if (!strcmp(argv[i], "--kv-q8")) {
            cache_type_alias = "q8_0";
            cfg->backend = QW3_BACKEND_METAL;
        } else if (!strcmp(argv[i], "--temp") && i + 1 < argc) {
            cfg->sample.temperature = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--sample-top-k") && i + 1 < argc) {
            cfg->sample.sample_top_k = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) {
            cfg->sample.top_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) {
            cfg->sample.min_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) {
            cfg->sample.repeat_penalty = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--repeat-last-n") && i + 1 < argc) {
            cfg->sample.repeat_last_n = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            cfg->sample.rng = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--dump-prompt")) {
            cfg->dump_prompt = true;
        } else if (!strcmp(argv[i], "--cpu")) {
            cfg->backend = QW3_BACKEND_CPU;
        } else if (!strcmp(argv[i], "--metal")) {
            cfg->backend = QW3_BACKEND_METAL;
        } else if (!strcmp(argv[i], "--nothink")) {
            cfg->think_mode = QW3_THINK_NONE;
        } else if (!strcmp(argv[i], "--store-dir") && i + 1 < argc) {
            cfg->store_dir = agent_strdup(argv[++i]);
        } else if (!strcmp(argv[i], "--conversation") && i + 1 < argc) {
            cfg->conversation = agent_strdup(argv[++i]);
        } else if (!strcmp(argv[i], "--chdir") && i + 1 < argc) {
            cfg->chdir_path = agent_strdup(argv[++i]);
        } else if (!strcmp(argv[i], "--max-tool-rounds") && i + 1 < argc) {
            cfg->max_tool_rounds = atoi(argv[++i]);
            if (cfg->max_tool_rounds < 1) cfg->max_tool_rounds = 1;
            if (cfg->max_tool_rounds > 128) cfg->max_tool_rounds = 128;
        } else if (!strcmp(argv[i], "--no-tools")) {
            cfg->tools_enabled = false;
        } else if (!strcmp(argv[i], "--tool-dsml") && i + 1 < argc) {
            cfg->tool_dsml = argv[++i];
        } else if (!strcmp(argv[i], "--tool-dsml-file") && i + 1 < argc) {
            const char *path = argv[++i];
            cfg->tool_dsml_owned = read_file_text(path, NULL);
            if (!cfg->tool_dsml_owned) {
                fprintf(stderr, "agent: cannot read DSML file %s\n", path);
                return -1;
            }
            cfg->tool_dsml = cfg->tool_dsml_owned;
        } else if (!strcmp(argv[i], "--tool-native") && i + 1 < argc) {
            cfg->tool_native = argv[++i];
        } else if (!strcmp(argv[i], "--tool-native-file") && i + 1 < argc) {
            const char *path = argv[++i];
            cfg->tool_native_owned = read_file_text(path, NULL);
            if (!cfg->tool_native_owned) {
                fprintf(stderr, "agent: cannot read native tool file %s\n", path);
                return -1;
            }
            cfg->tool_native = cfg->tool_native_owned;
        } else {
            fprintf(stderr, "agent: unknown option '%s'\n", argv[i]);
            print_help();
            return -1;
        }
    }
    if (!qw3_backend_supported(cfg->backend)) {
        fprintf(stderr, "agent: backend %s is not supported\n",
                qw3_backend_name(cfg->backend));
        return -1;
    }
    if (ngl_set) {
        if (cfg->backend != QW3_BACKEND_METAL) {
            fprintf(stderr, "agent: --ngl is available only with the Metal backend\n");
            return -1;
        }
        if (ngl < 0 || ngl > QW3_AGENT_N_LAYER) {
            fprintf(stderr, "agent: --ngl must be in the range 0..%d\n",
                    QW3_AGENT_N_LAYER);
            return -1;
        }
        char ngl_env[16];
        snprintf(ngl_env, sizeof(ngl_env), "%d", ngl);
        setenv("QW3_METAL_NGL", ngl_env, 1);
    }
    if (cache_type_alias) {
        if (cache_type_k || cache_type_v) {
            fprintf(stderr,
                    "agent: use either --kv-f16/--kv-f32/--kv-q8 or -ctk/-ctv, not both\n");
            return -1;
        }
        cache_type_k = cache_type_alias;
        cache_type_v = cache_type_alias;
    }
    if (cache_type_k || cache_type_v) {
        if (cfg->backend != QW3_BACKEND_METAL || !cache_type_k || !cache_type_v ||
            strcmp(cache_type_k, cache_type_v) != 0) {
            fprintf(stderr, "agent: -ctk/-ctv require matching Metal types\n");
            return -1;
        }
        if (!strcmp(cache_type_k, "q8_0")) {
            setenv("QW3_METAL_KV_Q8_0", "1", 1);
            setenv("QW3_METAL_KV_F16", "0", 1);
        } else if (!strcmp(cache_type_k, "f16")) {
            setenv("QW3_METAL_KV_Q8_0", "0", 1);
            setenv("QW3_METAL_KV_F16", "1", 1);
        } else if (!strcmp(cache_type_k, "f32")) {
            setenv("QW3_METAL_KV_Q8_0", "0", 1);
            setenv("QW3_METAL_KV_F16", "0", 1);
        } else {
            fprintf(stderr,
                    "agent: unsupported KV cache type '%s' (expected f32, f16, or q8_0)\n",
                    cache_type_k);
            return -1;
        }
    }
    if (!cfg->store_dir) cfg->store_dir = default_store_dir();
    if (!cfg->store_dir || ensure_dir(cfg->store_dir) != 0) {
        fprintf(stderr, "agent: cannot create store directory\n");
        return -1;
    }
    cfg->system_prompt = build_system_prompt(cfg->user_system, cfg->tools_enabled);
    if (!cfg->system_prompt) return -1;
    return 0;
}

static void free_config(agent_config *cfg) {
    free(cfg->prompt_owned);
    free(cfg->user_system_owned);
    free(cfg->system_prompt);
    free(cfg->store_dir);
    free(cfg->conversation);
    free(cfg->chdir_path);
    free(cfg->tool_dsml_owned);
    free(cfg->tool_native_owned);
}

static void agent_init_transcript(agent_state *a) {
    agent_clear_session_meta(a);
    qw3_tokens_free(&a->transcript);
    memset(&a->transcript, 0, sizeof(a->transcript));
    qw3_chat_append_message(a->engine, &a->transcript,
                            "system", a->cfg.system_prompt);
    if (a->session) qw3_session_invalidate(a->session);
}

static int agent_dump_rendered_prompt(agent_state *a, const char *user_prompt) {
    if (!a) return 1;
    qw3_tokens tmp;
    memset(&tmp, 0, sizeof(tmp));
    qw3_tokens_copy(&tmp, &a->transcript);
    if (user_prompt && user_prompt[0]) {
        qw3_chat_append_message(a->engine, &tmp, "user", user_prompt);
        qw3_chat_append_assistant_prefix(a->engine, &tmp, a->cfg.think_mode);
    }
    char *text = transcript_rendered_text(a, &tmp);
    if (!text) {
        qw3_tokens_free(&tmp);
        return 1;
    }
    fwrite(text, 1, strlen(text), stdout);
    if (text[0] && text[strlen(text) - 1] != '\n') fputc('\n', stdout);
    free(text);
    qw3_tokens_free(&tmp);
    return ferror(stdout) ? 1 : 0;
}

static void interactive_help(void) {
    fprintf(stderr,
        "Commands:\n"
        "  /help              Show this help\n"
        "  /quit              Exit\n"
        "  /new               Start a new conversation\n"
        "  /ctx               Print token count and context size\n"
        "  /save [name]       Save conversation\n"
        "  /list              List saved conversations\n"
        "  /switch id         Switch to a saved conversation\n"
        "  /del id            Delete a saved conversation\n"
        "  /strip [id]        Keep text/metadata and remove token payload\n"
        "  /read PATH         Send file content as the next message\n"
        "  /think             Enable thinking mode\n"
        "  /nothink           Disable thinking mode\n"
        "  /tools on|off      Enable or disable tool execution\n");
}

static char *read_line_agent(agent_state *a) {
    (void)a;
    if (isatty(STDIN_FILENO)) {
        char *line = linenoise("qw3-agent> ");
        if (line && line[0]) linenoiseHistoryAdd(line);
        return line;
    }
    char *line = NULL;
    size_t cap = 0;
    ssize_t n = getline(&line, &cap, stdin);
    if (n < 0) {
        free(line);
        return NULL;
    }
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
    return line;
}

static int handle_command(agent_state *a, char *line, char **message_out) {
    *message_out = NULL;
    if (line[0] != '/') {
        *message_out = agent_strdup(line);
        return 1;
    }
    if (!strcmp(line, "/help")) {
        interactive_help();
    } else if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) {
        return -1;
    } else if (!strcmp(line, "/new")) {
        agent_init_transcript(a);
        agent_statusf(a, "agent: new conversation\n");
    } else if (!strcmp(line, "/ctx")) {
        agent_statusf(a, "tokens=%d ctx=%d tools=%s think=%s\n",
                      a->transcript.len, a->cfg.ctx_size,
                      a->cfg.tools_enabled ? "on" : "off",
                      qw3_think_mode_name(a->cfg.think_mode));
    } else if (!strncmp(line, "/save", 5)) {
        const char *name = line[5] == ' ' ? line + 6 : a->cfg.conversation;
        if (name && !name[0]) name = NULL;
        store_save(a, name);
    } else if (!strcmp(line, "/list") || !strcmp(line, "/sessions")) {
        store_list(a);
    } else if (!strncmp(line, "/switch ", 8)) {
        if (a->session_id || a->session_title) (void)store_save(a, NULL);
        if (store_load(a, line + 8) != 0) {
            agent_statusf(a, "agent: cannot switch to %s\n", line + 8);
        }
    } else if (!strncmp(line, "/load ", 6)) {
        if (a->session_id || a->session_title) (void)store_save(a, NULL);
        if (store_load(a, line + 6) != 0) {
            agent_statusf(a, "agent: cannot load %s\n", line + 6);
        }
    } else if (!strncmp(line, "/del ", 5)) {
        (void)store_delete(a, line + 5);
    } else if (!strcmp(line, "/strip")) {
        (void)store_strip(a, NULL);
    } else if (!strncmp(line, "/strip ", 7)) {
        (void)store_strip(a, line + 7);
    } else if (!strncmp(line, "/read ", 6)) {
        char *text = read_file_text(line + 6, NULL);
        if (!text) {
            agent_statusf(a, "agent: cannot read %s: %s\n",
                          line + 6, strerror(errno));
        } else {
            *message_out = text;
            return 1;
        }
    } else if (!strcmp(line, "/think")) {
        a->cfg.think_mode = QW3_THINK_ON;
        agent_statusf(a, "agent: thinking enabled\n");
    } else if (!strcmp(line, "/nothink")) {
        a->cfg.think_mode = QW3_THINK_NONE;
        agent_statusf(a, "agent: thinking disabled\n");
    } else if (!strcmp(line, "/tools on")) {
        a->cfg.tools_enabled = true;
        agent_statusf(a, "agent: tools enabled\n");
    } else if (!strcmp(line, "/tools off")) {
        a->cfg.tools_enabled = false;
        agent_statusf(a, "agent: tools disabled\n");
    } else {
        agent_statusf(a, "agent: unknown command '%s'\n", line);
    }
    return 0;
}

typedef struct {
    char **v;
    int len;
    int cap;
} agent_input_queue;

typedef struct {
    agent_state *agent;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    int wake_rd;
    int wake_wr;
    bool stop;
    bool busy;
    bool has_job;
    bool interrupt;
    bool turn_done;
    int turn_rc;
    char *job;
    strbuf out;
    bool progress_active;
    int progress_current;
    int progress_total;
    double progress_tps;
} agent_worker;

static volatile sig_atomic_t g_agent_sigint = 0;

static void agent_sigint_handler(int sig) {
    (void)sig;
    g_agent_sigint = 1;
}

static int agent_set_nonblock(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static void agent_worker_wake(agent_worker *w) {
    if (!w || w->wake_wr < 0) return;
    char b = 1;
    ssize_t n = write(w->wake_wr, &b, 1);
    (void)n;
}

static void agent_worker_drain_wake(agent_worker *w) {
    char buf[64];
    while (w && w->wake_rd >= 0 && read(w->wake_rd, buf, sizeof(buf)) > 0) {
    }
}

static void agent_worker_output_write(void *ud, const char *s, size_t n) {
    agent_worker *w = (agent_worker *)ud;
    if (!w || !s || n == 0) return;
    pthread_mutex_lock(&w->mu);
    sb_append_n(&w->out, s, n);
    pthread_mutex_unlock(&w->mu);
    agent_worker_wake(w);
}

static bool agent_worker_should_interrupt(void *ud) {
    agent_worker *w = (agent_worker *)ud;
    if (!w) return false;
    pthread_mutex_lock(&w->mu);
    bool v = w->interrupt || w->stop;
    pthread_mutex_unlock(&w->mu);
    return v;
}

static bool agent_worker_is_busy(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool busy = w->busy || w->has_job;
    pthread_mutex_unlock(&w->mu);
    return busy;
}

static void agent_worker_progress_snapshot(agent_worker *w, bool *busy,
                                           bool *active, int *current,
                                           int *total, double *tps) {
    pthread_mutex_lock(&w->mu);
    if (busy) *busy = w->busy || w->has_job;
    if (active) *active = w->progress_active;
    if (current) *current = w->progress_current;
    if (total) *total = w->progress_total;
    if (tps) *tps = w->progress_tps;
    pthread_mutex_unlock(&w->mu);
}

static void agent_worker_progress_update(void *ud, const char *phase,
                                         int current, int total, double tps) {
    agent_worker *w = (agent_worker *)ud;
    if (!w || !phase) return;
    pthread_mutex_lock(&w->mu);
    if (!strcmp(phase, "prefill_done")) {
        w->progress_active = false;
        w->progress_current = current;
        w->progress_total = total;
        w->progress_tps = tps;
    } else if (!strcmp(phase, "prefill")) {
        w->progress_active = total > 0;
        w->progress_current = current;
        w->progress_total = total;
        w->progress_tps = tps;
    }
    pthread_mutex_unlock(&w->mu);
    agent_worker_wake(w);
}

static int agent_worker_submit(agent_worker *w, const char *message) {
    if (!w || !message) return -1;
    char *job = agent_strdup(message);
    if (!job) return -1;
    pthread_mutex_lock(&w->mu);
    if (w->busy || w->has_job || w->stop) {
        pthread_mutex_unlock(&w->mu);
        free(job);
        return -1;
    }
    w->job = job;
    w->has_job = true;
    w->interrupt = false;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
    agent_worker_wake(w);
    return 0;
}

static void agent_worker_interrupt(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    if (w->busy || w->has_job) w->interrupt = true;
    pthread_mutex_unlock(&w->mu);
    agent_worker_wake(w);
}

static void agent_worker_collect(agent_worker *w, char **out, size_t *out_len,
                                 bool *turn_done, int *turn_rc) {
    *out = NULL;
    if (out_len) *out_len = 0;
    *turn_done = false;
    *turn_rc = 0;
    pthread_mutex_lock(&w->mu);
    if (w->out.len > 0) {
        size_t n = w->out.len;
        *out = malloc(n + 1);
        if (*out) {
            memcpy(*out, w->out.p, n);
            (*out)[n] = '\0';
            if (out_len) *out_len = n;
            w->out.len = 0;
            if (w->out.p) w->out.p[0] = '\0';
        }
    }
    if (w->turn_done) {
        *turn_done = true;
        *turn_rc = w->turn_rc;
        w->turn_done = false;
    }
    pthread_mutex_unlock(&w->mu);
}

static void *agent_worker_main(void *ud) {
    agent_worker *w = (agent_worker *)ud;
    for (;;) {
        pthread_mutex_lock(&w->mu);
        while (!w->stop && !w->has_job) {
            pthread_cond_wait(&w->cond, &w->mu);
        }
        if (w->stop) {
            pthread_mutex_unlock(&w->mu);
            break;
        }
        char *job = w->job;
        w->job = NULL;
        w->has_job = false;
        w->busy = true;
        w->interrupt = false;
        pthread_mutex_unlock(&w->mu);

        int rc = run_agent_turn(w->agent, job);
        free(job);
        if (rc == 0 && w->agent->cfg.conversation) {
            store_save(w->agent, w->agent->cfg.conversation);
        }

        pthread_mutex_lock(&w->mu);
        w->busy = false;
        w->turn_done = true;
        w->turn_rc = rc;
        pthread_mutex_unlock(&w->mu);
        agent_worker_wake(w);
    }
    return NULL;
}

static int agent_worker_init(agent_worker *w, agent_state *a) {
    memset(w, 0, sizeof(*w));
    w->agent = a;
    w->wake_rd = -1;
    w->wake_wr = -1;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    sb_init(&w->out);
    int pfd[2];
    if (pipe(pfd) != 0) return -1;
    w->wake_rd = pfd[0];
    w->wake_wr = pfd[1];
    (void)agent_set_nonblock(w->wake_rd);
    (void)agent_set_nonblock(w->wake_wr);
    a->output_write = agent_worker_output_write;
    a->output_ud = w;
    a->status_write = agent_worker_output_write;
    a->status_ud = w;
    a->output_color = isatty(STDOUT_FILENO);
    a->should_interrupt = agent_worker_should_interrupt;
    a->interrupt_ud = w;
    a->progress_update = agent_worker_progress_update;
    a->progress_ud = w;
    if (pthread_create(&w->thread, NULL, agent_worker_main, w) != 0) {
        close(w->wake_rd);
        close(w->wake_wr);
        w->wake_rd = -1;
        w->wake_wr = -1;
        return -1;
    }
    return 0;
}

static void agent_worker_destroy(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    w->stop = true;
    w->interrupt = true;
    free(w->job);
    w->job = NULL;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
    agent_worker_wake(w);
    pthread_join(w->thread, NULL);
    if (w->wake_rd >= 0) close(w->wake_rd);
    if (w->wake_wr >= 0) close(w->wake_wr);
    sb_free(&w->out);
    pthread_cond_destroy(&w->cond);
    pthread_mutex_destroy(&w->mu);
}

static int agent_queue_push(agent_input_queue *q, const char *line) {
    if (q->len == q->cap) {
        int nc = q->cap ? q->cap * 2 : 8;
        char **nv = realloc(q->v, (size_t)nc * sizeof(*nv));
        if (!nv) return -1;
        q->v = nv;
        q->cap = nc;
    }
    q->v[q->len] = agent_strdup(line);
    if (!q->v[q->len]) return -1;
    q->len++;
    return 0;
}

static char *agent_queue_pop(agent_input_queue *q) {
    if (!q || q->len == 0) return NULL;
    char *out = q->v[0];
    memmove(q->v, q->v + 1, (size_t)(q->len - 1) * sizeof(*q->v));
    q->len--;
    return out;
}

static void agent_queue_free(agent_input_queue *q) {
    for (int i = 0; i < q->len; i++) free(q->v[i]);
    free(q->v);
    memset(q, 0, sizeof(*q));
}

static void agent_update_editor_status(struct linenoiseState *edit,
                                       agent_worker *w,
                                       const agent_input_queue *q) {
    bool busy = false;
    bool progress_active = false;
    int progress_current = 0;
    int progress_total = 0;
    double progress_tps = 0.0;
    agent_worker_progress_snapshot(w, &busy, &progress_active,
                                   &progress_current, &progress_total,
                                   &progress_tps);
    char status[160];
    if (busy && progress_active && progress_total > 0) {
        int pct = (int)((100.0 * (double)progress_current /
                         (double)progress_total) + 0.5);
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        char bar[19];
        int fill = (int)((18.0 * (double)progress_current /
                          (double)progress_total) + 0.5);
        if (fill < 0) fill = 0;
        if (fill > 18) fill = 18;
        for (int i = 0; i < 18; i++) bar[i] = i < fill ? '=' : '.';
        bar[18] = '\0';
        snprintf(status, sizeof(status),
                 "prefill [%s] %d/%d %d%% %.1f tok/s  queued=%d",
                 bar, progress_current, progress_total, pct, progress_tps,
                 q ? q->len : 0);
    } else if (busy) {
        snprintf(status, sizeof(status),
                 "state=running  queued=%d  ctx=busy  tools=%s  think=%s",
                 q ? q->len : 0,
                 w->agent->cfg.tools_enabled ? "on" : "off",
                 qw3_think_mode_name(w->agent->cfg.think_mode));
    } else {
        snprintf(status, sizeof(status),
                 "state=ready  queued=%d  ctx=%d/%d  tools=%s  think=%s",
                 q ? q->len : 0, w->agent->transcript.len,
                 w->agent->cfg.ctx_size,
                 w->agent->cfg.tools_enabled ? "on" : "off",
                 qw3_think_mode_name(w->agent->cfg.think_mode));
    }
    linenoiseEditSetStatus(edit, status, "\033[7m", "\033[0m");
}

static int agent_submit_or_queue(agent_worker *w, agent_input_queue *q,
                                 const char *message) {
    if (!agent_worker_is_busy(w) && q->len == 0) {
        return agent_worker_submit(w, message);
    }
    if (agent_queue_push(q, message) != 0) return -1;
    agent_statusf(w->agent, "agent: queued prompt (%d pending)\n", q->len);
    return 0;
}

static int agent_submit_next_if_idle(agent_worker *w, agent_input_queue *q) {
    if (agent_worker_is_busy(w) || q->len == 0) return 0;
    char *next = agent_queue_pop(q);
    if (!next) return 0;
    int rc = agent_worker_submit(w, next);
    free(next);
    return rc;
}

static void agent_display_worker_output(agent_worker *w,
                                        struct linenoiseState *edit,
                                        const agent_input_queue *q,
                                        bool *streaming_output,
                                        bool *turn_done, int *turn_rc) {
    char *out = NULL;
    size_t out_len = 0;
    bool done = false;
    int rc = 0;
    agent_worker_collect(w, &out, &out_len, &done, &rc);
    bool busy_now = agent_worker_is_busy(w);
    if (out) {
        if (edit && streaming_output && !*streaming_output) {
            linenoiseHide(edit);
        } else if (edit && !streaming_output) {
            linenoiseHide(edit);
        }
        agent_write_with_crlf(stdout, out, out_len);
        fflush(stdout);
        if (streaming_output && busy_now) *streaming_output = true;
        free(out);
    }
    if (!busy_now && streaming_output) *streaming_output = false;
    if (done && streaming_output) *streaming_output = false;
    if (edit) {
        if (!streaming_output || !*streaming_output) {
            agent_update_editor_status(edit, w, q);
            linenoiseShow(edit);
        }
    }
    if (done) {
        *turn_done = true;
        *turn_rc = rc;
    }
}

static int agent_process_interactive_line(agent_state *a, agent_worker *w,
                                          agent_input_queue *q, char *line) {
    if (!line || !line[0]) return 0;
    linenoiseHistoryAdd(line);
    if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) {
        return -1;
    }
    if (!strcmp(line, "/interrupt") || !strcmp(line, "/stop")) {
        agent_worker_interrupt(w);
        agent_statusf(a, "agent: interrupt requested\n");
        return 0;
    }
    if (line[0] == '/' && agent_worker_is_busy(w)) {
        agent_statusf(a, "agent: busy; use /interrupt or wait for the prompt\n");
        return 0;
    }
    if (line[0] == '/') {
        char *message = NULL;
        int action = handle_command(a, line, &message);
        if (action < 0) {
            free(message);
            return -1;
        }
        if (action == 1) {
            int rc = agent_submit_or_queue(w, q, message);
            free(message);
            return rc;
        }
        return 0;
    }
    return agent_submit_or_queue(w, q, line);
}

static int interactive_loop_worker(agent_state *a) {
    char *hist = path_join(a->cfg.store_dir, "history");
    if (hist) {
        linenoiseHistorySetMaxLen(1000);
        linenoiseHistoryLoad(hist);
    }

    agent_worker worker;
    if (agent_worker_init(&worker, a) != 0) {
        fprintf(stderr, "agent: cannot start worker thread\n");
        free(hist);
        return 1;
    }

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = agent_sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, &old_int);

    fprintf(stderr,
            "qw3-agent ready. /help for commands. /interrupt stops generation. store=%s\n",
            a->cfg.store_dir);

    int rc = 0;
    bool done = false;
    agent_input_queue queue = {0};
    struct linenoiseState edit;
    char editbuf[16384];
    bool edit_active = false;
    bool streaming_output = false;
    if (linenoiseEditStart(&edit, STDIN_FILENO, STDOUT_FILENO, editbuf,
                           sizeof(editbuf), "qw3-agent> ") != 0) {
        rc = 1;
        done = true;
    } else {
        edit_active = true;
        agent_update_editor_status(&edit, &worker, &queue);
        linenoiseShow(&edit);
    }

    while (!done) {
        struct pollfd fds[2];
        fds[0].fd = STDIN_FILENO;
        fds[0].events = POLLIN;
        fds[0].revents = 0;
        fds[1].fd = worker.wake_rd;
        fds[1].events = POLLIN;
        fds[1].revents = 0;
        int pr = poll(fds, 2, -1);
        if (g_agent_sigint) {
            g_agent_sigint = 0;
            if (agent_worker_is_busy(&worker)) {
                agent_worker_interrupt(&worker);
                agent_statusf(a, "\nagent: interrupt requested\n");
            } else {
                linenoiseEditClear(&edit);
            }
        }
        if (pr < 0) {
            if (errno == EINTR) continue;
            rc = 1;
            break;
        }
        if (fds[1].revents & POLLIN) {
            agent_worker_drain_wake(&worker);
        }

        bool turn_done = false;
        int turn_rc = 0;
        agent_display_worker_output(&worker, edit_active ? &edit : NULL,
                                    &queue, &streaming_output,
                                    &turn_done, &turn_rc);
        if (turn_done) {
            if (turn_rc != 0) rc = 1;
            if (agent_submit_next_if_idle(&worker, &queue) != 0) rc = 1;
            if (edit_active) {
                agent_update_editor_status(&edit, &worker, &queue);
                linenoiseShow(&edit);
            }
        }

        if (fds[0].revents & (POLLIN | POLLHUP)) {
            errno = 0;
            char *line = linenoiseEditFeed(&edit);
            if (line == linenoiseEditMore) {
                continue;
            }
            if (!line) {
                if (errno == EAGAIN) {
                    if (agent_worker_is_busy(&worker)) {
                        agent_worker_interrupt(&worker);
                        agent_statusf(a, "\nagent: interrupt requested\n");
                    }
                    linenoiseEditClear(&edit);
                    continue;
                }
                done = true;
                break;
            }

            linenoiseEditStop(&edit);
            edit_active = false;
            int action = agent_process_interactive_line(a, &worker, &queue, line);
            linenoiseFree(line);
            if (action < 0) {
                done = true;
                break;
            }

            turn_done = false;
            turn_rc = 0;
            agent_display_worker_output(&worker, NULL, &queue,
                                        &streaming_output,
                                        &turn_done, &turn_rc);
            if (turn_done) {
                if (turn_rc != 0) rc = 1;
                if (agent_submit_next_if_idle(&worker, &queue) != 0) rc = 1;
            }

            if (!done) {
                if (linenoiseEditStart(&edit, STDIN_FILENO, STDOUT_FILENO,
                                       editbuf, sizeof(editbuf),
                                       "qw3-agent> ") != 0) {
                    rc = 1;
                    done = true;
                } else {
                    edit_active = true;
                    if (!streaming_output) {
                        agent_update_editor_status(&edit, &worker, &queue);
                        linenoiseShow(&edit);
                    }
                }
            }
        }
    }

    if (edit_active) linenoiseEditStop(&edit);
    agent_queue_free(&queue);
    sigaction(SIGINT, &old_int, NULL);
    if (hist) {
        linenoiseHistorySave(hist);
        free(hist);
    }
    agent_worker_destroy(&worker);
    a->output_write = NULL;
    a->output_ud = NULL;
    a->status_write = NULL;
    a->status_ud = NULL;
    a->should_interrupt = NULL;
    a->interrupt_ud = NULL;
    a->progress_update = NULL;
    a->progress_ud = NULL;
    return rc;
}

static int interactive_loop_blocking(agent_state *a) {
    char *hist = path_join(a->cfg.store_dir, "history");
    if (hist) {
        linenoiseHistorySetMaxLen(1000);
        linenoiseHistoryLoad(hist);
    }
    fprintf(stderr, "qw3-agent ready. /help for commands. store=%s\n",
            a->cfg.store_dir);
    int rc = 0;
    for (;;) {
        char *line = read_line_agent(a);
        if (!line) break;
        if (!line[0]) {
            linenoiseFree(line);
            continue;
        }
        char *message = NULL;
        int action = handle_command(a, line, &message);
        linenoiseFree(line);
        if (action < 0) break;
        if (action == 0) continue;
        if (run_agent_turn(a, message) != 0) {
            free(message);
            rc = 1;
            break;
        }
        free(message);
        if (a->cfg.conversation) store_save(a, a->cfg.conversation);
    }
    if (hist) {
        linenoiseHistorySave(hist);
        free(hist);
    }
    return rc;
}

static int interactive_loop(agent_state *a) {
    if (isatty(STDIN_FILENO)) return interactive_loop_worker(a);
    return interactive_loop_blocking(a);
}

typedef struct {
    bool active;
} agent_direct_progress;

static bool agent_direct_progress_enabled(void) {
    const char *force = getenv("QW3_AGENT_PROGRESS");
    return (force && force[0] && strcmp(force, "0")) || isatty(STDERR_FILENO);
}

static void agent_direct_progress_update(void *ud, const char *phase,
                                         int current, int total, double tps) {
    agent_direct_progress *p = (agent_direct_progress *)ud;
    if (!p || !phase) return;
    if (!strcmp(phase, "prefill_done")) {
        if (p->active) {
            fprintf(stderr, "\n");
            fflush(stderr);
            p->active = false;
        }
        return;
    }
    if (strcmp(phase, "prefill") || total <= 0) return;
    int pct = (int)((100.0 * (double)current / (double)total) + 0.5);
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    char bar[25];
    int fill = (int)((24.0 * (double)current / (double)total) + 0.5);
    if (fill < 0) fill = 0;
    if (fill > 24) fill = 24;
    for (int i = 0; i < 24; i++) bar[i] = i < fill ? '=' : '.';
    bar[24] = '\0';
    fprintf(stderr, "\rprefill [%s] %d/%d %d%% %.1f tok/s",
            bar, current, total, pct, tps);
    fflush(stderr);
    p->active = true;
}

int main(int argc, char **argv) {
    agent_state a;
    memset(&a, 0, sizeof(a));
    if (parse_args(&a.cfg, argc, argv) != 0) {
        free_config(&a.cfg);
        return 1;
    }
    a.output_color = isatty(STDOUT_FILENO) || isatty(STDERR_FILENO);
    if (a.cfg.chdir_path && chdir(a.cfg.chdir_path) != 0) {
        fprintf(stderr, "agent: cannot chdir to %s: %s\n",
                a.cfg.chdir_path, strerror(errno));
        free_config(&a.cfg);
        return 1;
    }

    if (a.cfg.tool_dsml) {
        int rc = run_tool_dsml(&a, a.cfg.tool_dsml);
        agent_reset_source_read_budget(&a);
        free(a.last_read_path);
        free_config(&a.cfg);
        return rc;
    }
    if (a.cfg.tool_native) {
        int rc = run_tool_native(&a, a.cfg.tool_native);
        agent_reset_source_read_budget(&a);
        free(a.last_read_path);
        free_config(&a.cfg);
        return rc;
    }

    qw3_log(stderr, QW3_LOG_OK,
            "qw3-agent: model=%s backend=%s ctx=%d think=%s tools=%s\n",
            a.cfg.model_path, qw3_backend_name(a.cfg.backend), a.cfg.ctx_size,
            qw3_think_mode_name(a.cfg.think_mode),
            a.cfg.tools_enabled ? "on" : "off");
    qw3_context_memory mem = qw3_context_memory_estimate(a.cfg.backend,
                                                         a.cfg.ctx_size);
    qw3_log(stderr, QW3_LOG_TIMING,
            "qw3-agent: memory estimate %.1f MB\n",
            (double)mem.total_bytes / (1024.0 * 1024.0));

    qw3_engine_options opt = {
        .model_path = a.cfg.model_path,
        .backend = a.cfg.backend,
        .n_threads = 0,
        .warm_weights = false,
    };
    if (qw3_engine_open(&a.engine, &opt) != 0) {
        fprintf(stderr, "agent: engine open failed\n");
        free_config(&a.cfg);
        return 1;
    }
    if (qw3_session_create(&a.session, a.engine, a.cfg.ctx_size) != 0) {
        fprintf(stderr, "agent: cannot create %s session\n",
                qw3_backend_name(a.cfg.backend));
        qw3_engine_close(a.engine);
        free_config(&a.cfg);
        return 1;
    }

    agent_init_transcript(&a);
    if (a.cfg.conversation) {
        (void)store_load(&a, a.cfg.conversation);
    }
    if (a.cfg.dump_prompt) {
        int dump_rc = agent_dump_rendered_prompt(&a, a.cfg.prompt);
        agent_reset_source_read_budget(&a);
        free(a.last_read_path);
        agent_clear_session_meta(&a);
        qw3_tokens_free(&a.transcript);
        qw3_session_free(a.session);
        qw3_engine_close(a.engine);
        free_config(&a.cfg);
        return dump_rc;
    }

    int rc = 0;
    agent_direct_progress direct_progress = {0};
    if (a.cfg.prompt) {
        if (agent_direct_progress_enabled()) {
            a.progress_update = agent_direct_progress_update;
            a.progress_ud = &direct_progress;
        }
        rc = run_agent_turn(&a, a.cfg.prompt) == 0 ? 0 : 1;
        if (direct_progress.active) {
            fprintf(stderr, "\n");
            direct_progress.active = false;
        }
        if (rc == 0 && a.cfg.conversation) store_save(&a, a.cfg.conversation);
    } else {
        rc = interactive_loop(&a);
    }

    agent_reset_source_read_budget(&a);
    free(a.last_read_path);
    agent_clear_session_meta(&a);
    qw3_tokens_free(&a.transcript);
    qw3_session_free(a.session);
    qw3_engine_close(a.engine);
    free_config(&a.cfg);
    return rc;
}
