/* =========================================================================
 * qw3_agent.c - Qwen3 agent client with DSML tools and a lightweight store.
 * ========================================================================= */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
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

#define AGENT_STORE_MAGIC "QW3AGKV1"
#define AGENT_STORE_VERSION 1u

typedef struct {
    float temperature;
    int sample_top_k;
    float top_p;
    float min_p;
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
    const char *tool_dsml;
    char *tool_dsml_owned;
    int n_predict;
    int ctx_size;
    int max_tool_rounds;
    bool tools_enabled;
    qw3_backend backend;
    qw3_think_mode think_mode;
    sample_opts sample;
} agent_config;

typedef struct {
    qw3_engine *engine;
    qw3_session *session;
    qw3_tokens transcript;
    agent_config cfg;
    char *last_read_path;
    int last_read_next;
} agent_state;

typedef struct {
    qw3_engine *engine;
    qw3_tokens generated;
    strbuf text;
    size_t printed;
    bool printed_any;
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

static char *store_path(agent_state *a, const char *name) {
    char *safe = sanitized_name(name);
    if (!safe) return NULL;
    size_t n = strlen(safe) + 7;
    char *file = malloc(n);
    if (!file) {
        free(safe);
        return NULL;
    }
    snprintf(file, n, "%s.qw3a", safe);
    free(safe);
    char *path = path_join(a->cfg.store_dir, file);
    free(file);
    return path;
}

static int store_save(agent_state *a, const char *name) {
    char *path = store_path(a, name);
    if (!path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "agent: cannot save %s: %s\n", path, strerror(errno));
        free(path);
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
    if (rc == 0) {
        fprintf(stderr, "agent: saved %s (%d tokens)\n", path, a->transcript.len);
    } else {
        fprintf(stderr, "agent: failed while writing %s\n", path);
    }
    free(path);
    return rc;
}

static int store_load(agent_state *a, const char *name) {
    char *path = store_path(a, name);
    if (!path) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return -1;
    }
    agent_store_header h = {0};
    int rc = -1;
    if (fread(&h, 1, sizeof(h), fp) == sizeof(h) &&
        memcmp(h.magic, AGENT_STORE_MAGIC, sizeof(h.magic)) == 0 &&
        h.version == AGENT_STORE_VERSION &&
        h.token_len <= (uint32_t)a->cfg.ctx_size) {
        qw3_tokens next = {0};
        rc = 0;
        for (uint32_t i = 0; i < h.token_len; i++) {
            int32_t tok = 0;
            if (fread(&tok, 1, sizeof(tok), fp) != sizeof(tok)) {
                rc = -1;
                break;
            }
            qw3_tokens_push(&next, (int)tok);
        }
        if (rc == 0) {
            qw3_tokens_free(&a->transcript);
            a->transcript = next;
            if (a->session) qw3_session_invalidate(a->session);
            fprintf(stderr, "agent: loaded %s (%d tokens)\n",
                    path, a->transcript.len);
        } else {
            qw3_tokens_free(&next);
        }
    }
    fclose(fp);
    free(path);
    return rc;
}

static void store_list(agent_state *a) {
    DIR *dir = opendir(a->cfg.store_dir);
    if (!dir) {
        fprintf(stderr, "agent: cannot open store %s\n", a->cfg.store_dir);
        return;
    }
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        size_t n = strlen(de->d_name);
        if (n > 5 && strcmp(de->d_name + n - 5, ".qw3a") == 0) {
            fprintf(stderr, "  %.*s\n", (int)(n - 5), de->d_name);
        }
    }
    closedir(dir);
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

static int int_param(const tool_call *call, const char *name, int def) {
    const char *v = tool_param_value(call, name);
    return v && v[0] ? atoi(v) : def;
}

static char *tool_read(agent_state *a, const tool_call *call) {
    const char *path = tool_param_value(call, "path");
    if (!path || !path[0]) path = tool_param_value(call, "file");
    if (!path || !path[0]) return agent_strdup("error: read requires path");
    int start = int_param(call, "start", 1);
    int lines = int_param(call, "lines", 160);
    if (start < 1) start = 1;
    if (lines <= 0 || lines > 400) lines = 160;

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

static char *tool_write(const tool_call *call) {
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
    strbuf out;
    sb_init(&out);
    sb_printf(&out, "ok: wrote %s (%zu bytes)", path, strlen(content));
    return out.p;
}

static char *tool_edit(const tool_call *call) {
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
    return agent_strdup("ok: edited first occurrence");
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
    if (!strcmp(call->name, "write")) return tool_write(call);
    if (!strcmp(call->name, "edit")) return tool_edit(call);
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
        fprintf(stderr, "\n[tool] %s\n", call->name);
        char *out = execute_one_tool(a, call);
        sb_printf(&result, "<tool_result name=\"%s\">\n%s\n</tool_result>\n",
                  call->name, out ? out : "");
        fprintf(stderr, "%s\n", out ? out : "");
        free(out);
    }
    return result.p ? result.p : agent_strdup("");
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
        fputs(result, stdout);
        if (result[0] && result[strlen(result) - 1] != '\n') fputc('\n', stdout);
    }
    free(result);
    return 0;
}

static void agent_flush_visible(agent_emit_ctx *ctx, bool final) {
    const char *dsml = ctx->text.p ? strstr(ctx->text.p, DSML_BEGIN) : NULL;
    size_t limit = 0;
    if (dsml) {
        limit = (size_t)(dsml - ctx->text.p);
    } else if (final) {
        limit = ctx->text.len;
    } else {
        size_t hold = strlen(DSML_BEGIN);
        limit = ctx->text.len > hold ? ctx->text.len - hold : 0;
    }
    if (limit > ctx->printed) {
        fwrite(ctx->text.p + ctx->printed, 1, limit - ctx->printed, stdout);
        fflush(stdout);
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
    if (ctx->printed_any) {
        fputc('\n', stdout);
        fflush(stdout);
    }
}

static void append_generated_assistant(agent_state *a,
                                       const qw3_tokens *generated) {
    for (int i = 0; i < generated->len; i++) {
        qw3_tokens_push(&a->transcript, generated->v[i]);
    }
    int eos = qw3_token_eos(a->engine);
    if (eos >= 0 && (a->transcript.len == 0 ||
                     a->transcript.v[a->transcript.len - 1] != eos)) {
        qw3_tokens_push(&a->transcript, eos);
    }
}

static int generate_once(agent_state *a, char **assistant_text) {
    *assistant_text = NULL;
    qw3_chat_append_assistant_prefix(a->engine, &a->transcript,
                                     a->cfg.think_mode);

    agent_emit_ctx emit;
    memset(&emit, 0, sizeof(emit));
    emit.engine = a->engine;

    int rc = -1;
    if (a->cfg.backend == QW3_BACKEND_METAL) {
        if (a->cfg.sample.temperature > 0.0f) {
            rc = qw3_engine_metal_generate_sample(
                a->engine, &a->transcript, a->cfg.n_predict, a->cfg.ctx_size,
                a->cfg.sample.temperature, a->cfg.sample.sample_top_k,
                a->cfg.sample.top_p, a->cfg.sample.min_p,
                &a->cfg.sample.rng, agent_emit_token, agent_emit_done, &emit);
        } else {
            rc = qw3_engine_metal_generate_argmax(
                a->engine, &a->transcript, a->cfg.n_predict, a->cfg.ctx_size,
                agent_emit_token, agent_emit_done, &emit);
        }
    } else {
        char err[256] = {0};
        if (qw3_session_sync(a->session, &a->transcript, err, sizeof(err)) != 0) {
            fprintf(stderr, "agent: prefill failed: %s\n", err);
            rc = -1;
        } else {
            rc = 0;
            const int eos = qw3_token_eos(a->engine);
            for (int i = 0; i < a->cfg.n_predict; i++) {
                int token = qw3_session_sample(
                    a->session, a->cfg.sample.temperature,
                    a->cfg.sample.sample_top_k, a->cfg.sample.top_p,
                    a->cfg.sample.min_p, &a->cfg.sample.rng);
                if (token < 0) {
                    rc = -1;
                    break;
                }
                if (token == eos) break;
                agent_emit_token(&emit, token);
                if (qw3_session_eval(a->session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "agent: decode failed: %s\n", err);
                    rc = -1;
                    break;
                }
                if (emit.text.p && strstr(emit.text.p, DSML_END)) break;
            }
            agent_emit_done(&emit);
        }
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
    qw3_chat_append_message(a->engine, &a->transcript, "user", user_message);
    if (a->transcript.len >= a->cfg.ctx_size) {
        fprintf(stderr, "agent: context full (%d/%d tokens)\n",
                a->transcript.len, a->cfg.ctx_size);
        return -1;
    }

    for (int round = 0; round < a->cfg.max_tool_rounds; round++) {
        char *assistant = NULL;
        if (generate_once(a, &assistant) != 0) {
            free(assistant);
            return -1;
        }
        tool_call_list calls;
        int n_calls = parse_tool_calls(assistant ? assistant : "", &calls);
        free(assistant);
        if (n_calls <= 0) {
            free_tool_calls(&calls);
            return 0;
        }
        char *tool_result = execute_tools(a, &calls);
        free_tool_calls(&calls);
        qw3_chat_append_message(a->engine, &a->transcript, "tool", tool_result);
        free(tool_result);
    }
    fprintf(stderr, "agent: max tool rounds reached\n");
    return 0;
}

static char *build_system_prompt(const char *user_system) {
    strbuf sb;
    sb_init(&sb);
    sb_append(&sb,
        "You are qw3-agent, a coding and terminal assistant running inside "
        "the user's local project. Be concise, practical and careful with "
        "file changes. You can use tools when you need local information or "
        "need to edit files.\n\n"
        "When you need tools, answer only with a DSML block. After the tool "
        "result, continue normally. Tool format:\n"
        DSML_BEGIN "\n"
        DSML_INVOKE " name=\"read\">\n"
        DSML_PARAM " name=\"path\" string=\"true\">path/to/file"
        DSML_PARAM_CLOSE "\n"
        DSML_INVOKE_CLOSE "\n"
        DSML_END "\n\n"
        "Available tools:\n"
        "- read(path,start,lines): read numbered file lines.\n"
        "- more(path,lines): continue the last read.\n"
        "- list(path,depth,max): list files.\n"
        "- search(pattern,path,max): search text files.\n"
        "- write(path,content): overwrite a file.\n"
        "- edit(path,old,new): replace the first exact occurrence.\n"
        "- bash(cmd): run a shell command and return output.\n");
    if (user_system && user_system[0]) {
        sb_append(&sb, "\nUser system instructions:\n");
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
        "  -ctk TYPE -ctv TYPE  Metal KV cache type: f32 or q8_0\n"
        "  --temp N             Temperature (default: 0)\n"
        "  --sample-top-k N     Sampling top-k (default: 40)\n"
        "  --top-p N            Sampling top-p (default: 0.95)\n"
        "  --min-p N            Sampling min-p (default: 0)\n"
        "  --seed N             Sampling seed\n"
        "  --cpu                Use CPU backend\n"
        "  --metal              Use Metal backend\n"
        "  --nothink            Disable thinking mode\n"
        "  --store-dir PATH     Conversation store directory\n"
        "  --conversation NAME  Load/save a named conversation\n"
        "  --no-tools           Disable tool execution\n"
        "  --tool-dsml TEXT     Execute a literal DSML tool_calls block and exit\n"
        "  --tool-dsml-file PATH\n"
        "                       Execute DSML tool_calls read from a file and exit\n"
        "  --help               Show help\n\n"
        "Interactive commands:\n"
        "  /help, /quit, /new, /ctx, /save [name], /load name, /sessions\n"
        "  /read PATH, /think, /nothink, /tools on|off\n");
}

static int parse_args(agent_config *cfg, int argc, char **argv) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->model_path = "./qw3.gguf";
    cfg->n_predict = 768;
    cfg->ctx_size = 32768;
    cfg->max_tool_rounds = 8;
    cfg->tools_enabled = true;
    cfg->think_mode = QW3_THINK_ON;
    cfg->sample.temperature = 0.0f;
    cfg->sample.sample_top_k = 40;
    cfg->sample.top_p = 0.95f;
    cfg->sample.min_p = 0.0f;
    cfg->sample.rng = 0x123456789abcdef0ull;
#ifdef QW3_NO_METAL
    cfg->backend = QW3_BACKEND_CPU;
#else
    cfg->backend = QW3_BACKEND_METAL;
#endif
    const char *cache_type_k = NULL;
    const char *cache_type_v = NULL;
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
        } else if ((!strcmp(argv[i], "-ctk") || !strcmp(argv[i], "--ctk")) &&
                   i + 1 < argc) {
            cache_type_k = argv[++i];
            cfg->backend = QW3_BACKEND_METAL;
        } else if ((!strcmp(argv[i], "-ctv") || !strcmp(argv[i], "--ctv")) &&
                   i + 1 < argc) {
            cache_type_v = argv[++i];
            cfg->backend = QW3_BACKEND_METAL;
        } else if (!strcmp(argv[i], "--temp") && i + 1 < argc) {
            cfg->sample.temperature = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--sample-top-k") && i + 1 < argc) {
            cfg->sample.sample_top_k = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) {
            cfg->sample.top_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) {
            cfg->sample.min_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            cfg->sample.rng = strtoull(argv[++i], NULL, 10);
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
    if (cache_type_k || cache_type_v) {
        if (cfg->backend != QW3_BACKEND_METAL || !cache_type_k || !cache_type_v ||
            strcmp(cache_type_k, cache_type_v) != 0) {
            fprintf(stderr, "agent: -ctk/-ctv require matching Metal types\n");
            return -1;
        }
        if (!strcmp(cache_type_k, "q8_0")) {
            setenv("QW3_METAL_KV_Q8_0", "1", 1);
        } else if (!strcmp(cache_type_k, "f32")) {
            setenv("QW3_METAL_KV_Q8_0", "0", 1);
        } else {
            fprintf(stderr, "agent: unsupported KV cache type '%s'\n", cache_type_k);
            return -1;
        }
    }
    if (!cfg->store_dir) cfg->store_dir = default_store_dir();
    if (!cfg->store_dir || ensure_dir(cfg->store_dir) != 0) {
        fprintf(stderr, "agent: cannot create store directory\n");
        return -1;
    }
    cfg->system_prompt = build_system_prompt(cfg->user_system);
    if (!cfg->system_prompt) return -1;
    return 0;
}

static void free_config(agent_config *cfg) {
    free(cfg->prompt_owned);
    free(cfg->user_system_owned);
    free(cfg->system_prompt);
    free(cfg->store_dir);
    free(cfg->conversation);
    free(cfg->tool_dsml_owned);
}

static void agent_init_transcript(agent_state *a) {
    qw3_tokens_free(&a->transcript);
    memset(&a->transcript, 0, sizeof(a->transcript));
    qw3_chat_append_message(a->engine, &a->transcript,
                            "system", a->cfg.system_prompt);
    if (a->session) qw3_session_invalidate(a->session);
}

static void interactive_help(void) {
    fprintf(stderr,
        "Commands:\n"
        "  /help              Show this help\n"
        "  /quit              Exit\n"
        "  /new               Start a new conversation\n"
        "  /ctx               Print token count and context size\n"
        "  /save [name]       Save conversation\n"
        "  /load name         Load conversation\n"
        "  /sessions          List saved conversations\n"
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
        fprintf(stderr, "agent: new conversation\n");
    } else if (!strcmp(line, "/ctx")) {
        fprintf(stderr, "tokens=%d ctx=%d tools=%s think=%s\n",
                a->transcript.len, a->cfg.ctx_size,
                a->cfg.tools_enabled ? "on" : "off",
                qw3_think_mode_name(a->cfg.think_mode));
    } else if (!strncmp(line, "/save", 5)) {
        const char *name = line[5] == ' ' ? line + 6 : a->cfg.conversation;
        if (!name || !name[0]) name = "default";
        store_save(a, name);
    } else if (!strncmp(line, "/load ", 6)) {
        if (store_load(a, line + 6) != 0) {
            fprintf(stderr, "agent: cannot load %s\n", line + 6);
        }
    } else if (!strcmp(line, "/sessions")) {
        store_list(a);
    } else if (!strncmp(line, "/read ", 6)) {
        char *text = read_file_text(line + 6, NULL);
        if (!text) {
            fprintf(stderr, "agent: cannot read %s: %s\n",
                    line + 6, strerror(errno));
        } else {
            *message_out = text;
            return 1;
        }
    } else if (!strcmp(line, "/think")) {
        a->cfg.think_mode = QW3_THINK_ON;
        fprintf(stderr, "agent: thinking enabled\n");
    } else if (!strcmp(line, "/nothink")) {
        a->cfg.think_mode = QW3_THINK_NONE;
        fprintf(stderr, "agent: thinking disabled\n");
    } else if (!strcmp(line, "/tools on")) {
        a->cfg.tools_enabled = true;
        fprintf(stderr, "agent: tools enabled\n");
    } else if (!strcmp(line, "/tools off")) {
        a->cfg.tools_enabled = false;
        fprintf(stderr, "agent: tools disabled\n");
    } else {
        fprintf(stderr, "agent: unknown command '%s'\n", line);
    }
    return 0;
}

static int interactive_loop(agent_state *a) {
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

int main(int argc, char **argv) {
    agent_state a;
    memset(&a, 0, sizeof(a));
    if (parse_args(&a.cfg, argc, argv) != 0) {
        free_config(&a.cfg);
        return 1;
    }

    if (a.cfg.tool_dsml) {
        int rc = run_tool_dsml(&a, a.cfg.tool_dsml);
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
    if (a.cfg.backend != QW3_BACKEND_METAL &&
        qw3_session_create(&a.session, a.engine, a.cfg.ctx_size) != 0) {
        fprintf(stderr, "agent: cannot create CPU session\n");
        qw3_engine_close(a.engine);
        free_config(&a.cfg);
        return 1;
    }

    agent_init_transcript(&a);
    if (a.cfg.conversation) {
        (void)store_load(&a, a.cfg.conversation);
    }

    int rc = 0;
    if (a.cfg.prompt) {
        rc = run_agent_turn(&a, a.cfg.prompt) == 0 ? 0 : 1;
        if (rc == 0 && a.cfg.conversation) store_save(&a, a.cfg.conversation);
    } else {
        rc = interactive_loop(&a);
    }

    free(a.last_read_path);
    qw3_tokens_free(&a.transcript);
    qw3_session_free(a.session);
    qw3_engine_close(a.engine);
    free_config(&a.cfg);
    return rc;
}
