#define _POSIX_C_SOURCE 200809L

#include "qw3.h"
#ifndef QW3_NO_METAL
#include "qw3_vision.h"
#endif

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define QW3_SRV_MAX_BODY_BYTES (64u * 1024u * 1024u)
#define QW3_SRV_MAX_HEADER_BYTES (64u * 1024u)
#define QW3_SRV_MAX_STOPS 8
#define QW3_SRV_MAX_TOOL_CALLS 16
#define QW3_SRV_VIS_CACHE_CAP 8

static volatile sig_atomic_t g_stop_requested = 0;

static void srv_handle_signal(int sig) {
    (void)sig;
    g_stop_requested = 1;
}

static double srv_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

/* =========================================================================
 * Dynamic string buffer + JSON escaping
 * ========================================================================= */

typedef struct {
    char *p;
    size_t len;
    size_t cap;
} strbuf;

static void sb_init(strbuf *sb) {
    sb->p = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static void sb_free(strbuf *sb) {
    free(sb->p);
    sb->p = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static bool sb_reserve(strbuf *sb, size_t extra) {
    if (sb->len + extra + 1 <= sb->cap) return true;
    size_t ncap = sb->cap ? sb->cap * 2 : 256;
    while (ncap < sb->len + extra + 1) ncap *= 2;
    char *np = (char *)realloc(sb->p, ncap);
    if (!np) return false;
    sb->p = np;
    sb->cap = ncap;
    return true;
}

static void sb_append_n(strbuf *sb, const char *s, size_t n) {
    if (!s || n == 0) return;
    if (!sb_reserve(sb, n)) return;
    memcpy(sb->p + sb->len, s, n);
    sb->len += n;
    sb->p[sb->len] = '\0';
}

static void sb_append(strbuf *sb, const char *s) {
    if (s) sb_append_n(sb, s, strlen(s));
}

static void sb_append_c(strbuf *sb, char c) {
    sb_append_n(sb, &c, 1);
}

static void sb_appendf(strbuf *sb, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need > 0 && sb_reserve(sb, (size_t)need)) {
        vsnprintf(sb->p + sb->len, (size_t)need + 1, fmt, ap2);
        sb->len += (size_t)need;
    }
    va_end(ap2);
}

static void sb_append_json_str_n(strbuf *sb, const char *s, size_t n) {
    sb_append_c(sb, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  sb_append_n(sb, "\\\"", 2); break;
        case '\\': sb_append_n(sb, "\\\\", 2); break;
        case '\b': sb_append_n(sb, "\\b", 2); break;
        case '\f': sb_append_n(sb, "\\f", 2); break;
        case '\n': sb_append_n(sb, "\\n", 2); break;
        case '\r': sb_append_n(sb, "\\r", 2); break;
        case '\t': sb_append_n(sb, "\\t", 2); break;
        default:
            if (c < 0x20u) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                sb_append_n(sb, buf, 6);
            } else {
                sb_append_c(sb, (char)c);
            }
            break;
        }
    }
    sb_append_c(sb, '"');
}

static void sb_append_json_str(strbuf *sb, const char *s) {
    sb_append_json_str_n(sb, s ? s : "", s ? strlen(s) : 0);
}

static void sb_append_utf8(strbuf *sb, uint32_t cp) {
    char buf[4];
    if (cp <= 0x7Fu) {
        buf[0] = (char)cp;
        sb_append_n(sb, buf, 1);
    } else if (cp <= 0x7FFu) {
        buf[0] = (char)(0xC0u | (cp >> 6u));
        buf[1] = (char)(0x80u | (cp & 0x3Fu));
        sb_append_n(sb, buf, 2);
    } else if (cp <= 0xFFFFu) {
        buf[0] = (char)(0xE0u | (cp >> 12u));
        buf[1] = (char)(0x80u | ((cp >> 6u) & 0x3Fu));
        buf[2] = (char)(0x80u | (cp & 0x3Fu));
        sb_append_n(sb, buf, 3);
    } else if (cp <= 0x10FFFFu) {
        buf[0] = (char)(0xF0u | (cp >> 18u));
        buf[1] = (char)(0x80u | ((cp >> 12u) & 0x3Fu));
        buf[2] = (char)(0x80u | ((cp >> 6u) & 0x3Fu));
        buf[3] = (char)(0x80u | (cp & 0x3Fu));
        sb_append_n(sb, buf, 4);
    }
}

/* =========================================================================
 * Minimal recursive-descent JSON DOM parser (with raw slice preservation)
 * ========================================================================= */

typedef enum {
    J_NULL,
    J_BOOL,
    J_NUM,
    J_STR,
    J_ARR,
    J_OBJ,
} jtype;

typedef struct jval jval;

typedef struct {
    char *key;
    jval *val;
} jpair;

struct jval {
    jtype type;
    const char *raw;
    size_t raw_len;
    bool b;
    double num;
    char *str;
    size_t str_len;
    jval **items;
    jpair *pairs;
    size_t len;
    size_t cap;
};

static void jval_free(jval *v) {
    if (!v) return;
    if (v->type == J_STR) {
        free(v->str);
    } else if (v->type == J_ARR) {
        for (size_t i = 0; i < v->len; i++) jval_free(v->items[i]);
        free(v->items);
    } else if (v->type == J_OBJ) {
        for (size_t i = 0; i < v->len; i++) {
            free(v->pairs[i].key);
            jval_free(v->pairs[i].val);
        }
        free(v->pairs);
    }
    free(v);
}

typedef struct {
    const char *p;
    const char *end;
    int depth;
} jparser;

static void jp_skip_ws(jparser *jp) {
    while (jp->p < jp->end && (*jp->p == ' ' || *jp->p == '\t' || *jp->p == '\r' || *jp->p == '\n'))
        jp->p++;
}

static int jp_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool jp_parse_hex4(jparser *jp, uint32_t *out) {
    if (jp->end - jp->p < 4) return false;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        int h = jp_hex(jp->p[i]);
        if (h < 0) return false;
        v = (v << 4u) | (uint32_t)h;
    }
    jp->p += 4;
    *out = v;
    return true;
}

static char *jp_parse_string_raw(jparser *jp, size_t *out_len) {
    if (jp->p >= jp->end || *jp->p != '"') return NULL;
    jp->p++;
    strbuf sb;
    sb_init(&sb);
    sb_reserve(&sb, 32);
    while (jp->p < jp->end) {
        char c = *jp->p++;
        if (c == '"') {
            if (!sb.p) {
                sb.p = (char *)calloc(1, 1);
                sb.len = 0;
            }
            if (out_len) *out_len = sb.len;
            return sb.p;
        }
        if ((unsigned char)c < 0x20u) {
            sb_free(&sb);
            return NULL;
        }
        if (c == '\\') {
            if (jp->p >= jp->end) { sb_free(&sb); return NULL; }
            char e = *jp->p++;
            switch (e) {
            case '"':  sb_append_c(&sb, '"'); break;
            case '\\': sb_append_c(&sb, '\\'); break;
            case '/':  sb_append_c(&sb, '/'); break;
            case 'b':  sb_append_c(&sb, '\b'); break;
            case 'f':  sb_append_c(&sb, '\f'); break;
            case 'n':  sb_append_c(&sb, '\n'); break;
            case 'r':  sb_append_c(&sb, '\r'); break;
            case 't':  sb_append_c(&sb, '\t'); break;
            case 'u': {
                uint32_t u1 = 0;
                if (!jp_parse_hex4(jp, &u1)) { sb_free(&sb); return NULL; }
                if (u1 >= 0xD800u && u1 <= 0xDBFFu && jp->end - jp->p >= 6 &&
                    jp->p[0] == '\\' && jp->p[1] == 'u') {
                    jp->p += 2;
                    uint32_t u2 = 0;
                    if (!jp_parse_hex4(jp, &u2) || u2 < 0xDC00u || u2 > 0xDFFFu) {
                        sb_free(&sb);
                        return NULL;
                    }
                    uint32_t cp = 0x10000u + (((u1 - 0xD800u) << 10u) | (u2 - 0xDC00u));
                    sb_append_utf8(&sb, cp);
                } else {
                    sb_append_utf8(&sb, u1);
                }
                break;
            }
            default:
                sb_free(&sb);
                return NULL;
            }
        } else {
            sb_append_c(&sb, c);
        }
    }
    sb_free(&sb);
    return NULL;
}

static jval *jp_parse_value(jparser *jp);

static jval *jp_parse_array(jparser *jp) {
    const char *start = jp->p;
    jp->p++; /* '[' */
    jval *v = (jval *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->type = J_ARR;
    v->raw = start;
    jp_skip_ws(jp);
    if (jp->p < jp->end && *jp->p == ']') {
        jp->p++;
        v->raw_len = (size_t)(jp->p - start);
        return v;
    }
    while (jp->p < jp->end) {
        jp_skip_ws(jp);
        jval *item = jp_parse_value(jp);
        if (!item) { jval_free(v); return NULL; }
        if (v->len == v->cap) {
            size_t ncap = v->cap ? v->cap * 2 : 8;
            jval **ni = (jval **)realloc(v->items, ncap * sizeof(*ni));
            if (!ni) { jval_free(item); jval_free(v); return NULL; }
            v->items = ni;
            v->cap = ncap;
        }
        v->items[v->len++] = item;
        jp_skip_ws(jp);
        if (jp->p < jp->end && *jp->p == ',') {
            jp->p++;
            continue;
        }
        if (jp->p < jp->end && *jp->p == ']') {
            jp->p++;
            v->raw_len = (size_t)(jp->p - start);
            return v;
        }
        break;
    }
    jval_free(v);
    return NULL;
}

static jval *jp_parse_object(jparser *jp) {
    const char *start = jp->p;
    jp->p++; /* '{' */
    jval *v = (jval *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->type = J_OBJ;
    v->raw = start;
    jp_skip_ws(jp);
    if (jp->p < jp->end && *jp->p == '}') {
        jp->p++;
        v->raw_len = (size_t)(jp->p - start);
        return v;
    }
    while (jp->p < jp->end) {
        jp_skip_ws(jp);
        char *key = jp_parse_string_raw(jp, NULL);
        if (!key) { jval_free(v); return NULL; }
        jp_skip_ws(jp);
        if (jp->p >= jp->end || *jp->p != ':') {
            free(key);
            jval_free(v);
            return NULL;
        }
        jp->p++;
        jp_skip_ws(jp);
        jval *val = jp_parse_value(jp);
        if (!val) { free(key); jval_free(v); return NULL; }
        if (v->len == v->cap) {
            size_t ncap = v->cap ? v->cap * 2 : 8;
            jpair *np = (jpair *)realloc(v->pairs, ncap * sizeof(*np));
            if (!np) { free(key); jval_free(val); jval_free(v); return NULL; }
            v->pairs = np;
            v->cap = ncap;
        }
        v->pairs[v->len].key = key;
        v->pairs[v->len].val = val;
        v->len++;
        jp_skip_ws(jp);
        if (jp->p < jp->end && *jp->p == ',') {
            jp->p++;
            continue;
        }
        if (jp->p < jp->end && *jp->p == '}') {
            jp->p++;
            v->raw_len = (size_t)(jp->p - start);
            return v;
        }
        break;
    }
    jval_free(v);
    return NULL;
}

static jval *jp_parse_value(jparser *jp) {
    if (jp->depth > 64) return NULL;
    jp_skip_ws(jp);
    if (jp->p >= jp->end) return NULL;
    const char *start = jp->p;
    char c = *jp->p;
    if (c == '{') {
        jp->depth++;
        jval *v = jp_parse_object(jp);
        jp->depth--;
        return v;
    }
    if (c == '[') {
        jp->depth++;
        jval *v = jp_parse_array(jp);
        jp->depth--;
        return v;
    }
    if (c == '"') {
        size_t slen = 0;
        char *s = jp_parse_string_raw(jp, &slen);
        if (!s) return NULL;
        jval *v = (jval *)calloc(1, sizeof(*v));
        if (!v) { free(s); return NULL; }
        v->type = J_STR;
        v->raw = start;
        v->raw_len = (size_t)(jp->p - start);
        v->str = s;
        v->str_len = slen;
        return v;
    }
    if (c == 't' && jp->end - jp->p >= 4 && !memcmp(jp->p, "true", 4)) {
        jp->p += 4;
        jval *v = (jval *)calloc(1, sizeof(*v));
        if (!v) return NULL;
        v->type = J_BOOL;
        v->b = true;
        v->raw = start;
        v->raw_len = 4;
        return v;
    }
    if (c == 'f' && jp->end - jp->p >= 5 && !memcmp(jp->p, "false", 5)) {
        jp->p += 5;
        jval *v = (jval *)calloc(1, sizeof(*v));
        if (!v) return NULL;
        v->type = J_BOOL;
        v->b = false;
        v->raw = start;
        v->raw_len = 5;
        return v;
    }
    if (c == 'n' && jp->end - jp->p >= 4 && !memcmp(jp->p, "null", 4)) {
        jp->p += 4;
        jval *v = (jval *)calloc(1, sizeof(*v));
        if (!v) return NULL;
        v->type = J_NULL;
        v->raw = start;
        v->raw_len = 4;
        return v;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *ep = NULL;
        double d = strtod(jp->p, &ep);
        if (!ep || ep == jp->p || ep > jp->end) return NULL;
        jp->p = ep;
        jval *v = (jval *)calloc(1, sizeof(*v));
        if (!v) return NULL;
        v->type = J_NUM;
        v->num = d;
        v->raw = start;
        v->raw_len = (size_t)(jp->p - start);
        return v;
    }
    return NULL;
}

static jval *json_parse(const char *src, size_t len) {
    if (!src) return NULL;
    jparser jp = {src, src + len, 0};
    jval *v = jp_parse_value(&jp);
    if (!v) return NULL;
    jp_skip_ws(&jp);
    if (jp.p != jp.end) {
        jval_free(v);
        return NULL;
    }
    return v;
}

static const jval *jobj_get(const jval *obj, const char *key) {
    if (!obj || obj->type != J_OBJ || !key) return NULL;
    for (size_t i = 0; i < obj->len; i++) {
        if (strcmp(obj->pairs[i].key, key) == 0) return obj->pairs[i].val;
    }
    return NULL;
}

static const char *jobj_str(const jval *obj, const char *key) {
    const jval *v = jobj_get(obj, key);
    return (v && v->type == J_STR) ? v->str : NULL;
}

static bool jobj_bool(const jval *obj, const char *key, bool def) {
    const jval *v = jobj_get(obj, key);
    return (v && v->type == J_BOOL) ? v->b : def;
}

static double jobj_num(const jval *obj, const char *key, double def) {
    const jval *v = jobj_get(obj, key);
    return (v && v->type == J_NUM) ? v->num : def;
}

/* =========================================================================
 * Socket / HTTP helpers
 * ========================================================================= */

static bool send_all(int fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, buf + sent, len - sent, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        sent += (size_t)n;
    }
    return true;
}

static bool client_disconnected(int fd) {
    if (g_stop_requested) return true;
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN | POLLHUP | POLLERR;
    pfd.revents = 0;
    int r = poll(&pfd, 1, 0);
    if (r < 0) return errno != EINTR;
    if (r == 0) return false;
    if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) return true;
    if (pfd.revents & POLLIN) {
        char c;
        int flags = MSG_PEEK;
#ifdef MSG_DONTWAIT
        flags |= MSG_DONTWAIT;
#endif
        ssize_t n = recv(fd, &c, 1, flags);
        if (n == 0) return true;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return true;
    }
    return false;
}

static bool http_interrupt_cb(void *ud) {
    int fd = *(int *)ud;
    return client_disconnected(fd);
}

static void send_http_json(int fd, int status, const char *status_text,
                           const char *json, size_t len) {
    char hdr[512];
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, len);
    if (hlen > 0) {
        if (send_all(fd, hdr, (size_t)hlen) && len > 0) {
            (void)send_all(fd, json, len);
        }
    }
}

static void send_http_error(int fd, int status, const char *status_text,
                            const char *type, const char *msg) {
    strbuf sb;
    sb_init(&sb);
    sb_append(&sb, "{\"error\":{\"message\":");
    sb_append_json_str(&sb, msg ? msg : "error");
    sb_append(&sb, ",\"type\":");
    sb_append_json_str(&sb, type ? type : "invalid_request_error");
    sb_append(&sb, "}}\n");
    send_http_json(fd, status, status_text, sb.p, sb.len);
    sb_free(&sb);
}

static bool send_sse_headers(int fd) {
    static const char hdr[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "X-Accel-Buffering: no\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n"
        "\r\n";
    return send_all(fd, hdr, sizeof(hdr) - 1);
}

static bool send_sse_data(int fd, const char *json, size_t len) {
    if (!send_all(fd, "data: ", 6)) return false;
    if (!send_all(fd, json, len)) return false;
    return send_all(fd, "\n\n", 2);
}

typedef struct {
    char method[16];
    char path[256];
    char auth[256];
    char *body;
    size_t body_len;
} http_request;

static void http_request_free(http_request *req) {
    free(req->body);
    req->body = NULL;
    req->body_len = 0;
}

static int read_http_request(int fd, http_request *req) {
    memset(req, 0, sizeof(*req));
    strbuf raw;
    sb_init(&raw);
    char tmp[4096];
    size_t header_end = 0;
    while (raw.len < QW3_SRV_MAX_HEADER_BYTES) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            sb_free(&raw);
            return -1;
        }
        if (n == 0) {
            sb_free(&raw);
            return -1;
        }
        sb_append_n(&raw, tmp, (size_t)n);
        char *sep = strstr(raw.p, "\r\n\r\n");
        if (sep) {
            header_end = (size_t)(sep - raw.p) + 4;
            break;
        }
    }
    if (header_end == 0) {
        sb_free(&raw);
        return -2;
    }

    /* Parse request line */
    char *line_end = strstr(raw.p, "\r\n");
    if (!line_end) { sb_free(&raw); return -2; }
    char *sp1 = memchr(raw.p, ' ', (size_t)(line_end - raw.p));
    if (!sp1) { sb_free(&raw); return -2; }
    size_t mlen = (size_t)(sp1 - raw.p);
    if (mlen >= sizeof(req->method)) mlen = sizeof(req->method) - 1;
    memcpy(req->method, raw.p, mlen);
    req->method[mlen] = '\0';

    char *pstart = sp1 + 1;
    char *sp2 = memchr(pstart, ' ', (size_t)(line_end - pstart));
    if (!sp2) { sb_free(&raw); return -2; }
    size_t plen = (size_t)(sp2 - pstart);
    char *qmark = memchr(pstart, '?', plen);
    if (qmark) plen = (size_t)(qmark - pstart);
    if (plen >= sizeof(req->path)) plen = sizeof(req->path) - 1;
    memcpy(req->path, pstart, plen);
    req->path[plen] = '\0';

    /* Parse headers */
    size_t content_length = 0;
    bool chunked = false;
    char *cur = line_end + 2;
    while (cur < raw.p + header_end - 2) {
        char *eol = strstr(cur, "\r\n");
        if (!eol || eol == cur) break;
        char *colon = memchr(cur, ':', (size_t)(eol - cur));
        if (colon) {
            size_t klen = (size_t)(colon - cur);
            char *val = colon + 1;
            while (val < eol && (*val == ' ' || *val == '\t')) val++;
            size_t vlen = (size_t)(eol - val);
            while (vlen > 0 && (val[vlen - 1] == ' ' || val[vlen - 1] == '\t')) vlen--;
            if (klen == 14 && strncasecmp(cur, "Content-Length", 14) == 0) {
                char nbuf[32];
                size_t cl = vlen < sizeof(nbuf) - 1 ? vlen : sizeof(nbuf) - 1;
                memcpy(nbuf, val, cl);
                nbuf[cl] = '\0';
                content_length = (size_t)strtoull(nbuf, NULL, 10);
            } else if (klen == 13 && strncasecmp(cur, "Authorization", 13) == 0) {
                size_t al = vlen < sizeof(req->auth) - 1 ? vlen : sizeof(req->auth) - 1;
                memcpy(req->auth, val, al);
                req->auth[al] = '\0';
            } else if (klen == 17 && strncasecmp(cur, "Transfer-Encoding", 17) == 0) {
                if (vlen >= 7 && strncasecmp(val, "chunked", 7) == 0) chunked = true;
            }
        }
        cur = eol + 2;
    }

    if (chunked) {
        sb_free(&raw);
        return -3;
    }
    if (content_length > QW3_SRV_MAX_BODY_BYTES) {
        sb_free(&raw);
        return -4;
    }

    size_t have = raw.len - header_end;
    if (content_length > 0) {
        req->body = (char *)malloc(content_length + 1);
        if (!req->body) { sb_free(&raw); return -1; }
        size_t copy_now = have < content_length ? have : content_length;
        if (copy_now > 0) memcpy(req->body, raw.p + header_end, copy_now);
        size_t got = copy_now;
        while (got < content_length) {
            ssize_t n = recv(fd, req->body + got, content_length - got, 0);
            if (n < 0) {
                if (errno == EINTR) continue;
                free(req->body);
                req->body = NULL;
                sb_free(&raw);
                return -1;
            }
            if (n == 0) {
                free(req->body);
                req->body = NULL;
                sb_free(&raw);
                return -1;
            }
            got += (size_t)n;
        }
        req->body[content_length] = '\0';
        req->body_len = content_length;
    }
    sb_free(&raw);
    return 0;
}

/* =========================================================================
 * Server configuration & state
 * ========================================================================= */

typedef struct {
    const char *model_path;
    const char *mmproj_path;
    int vision_max_tokens;
    char model_alias[128];
    const char *host;
    int port;
    int ctx_size;
    qw3_backend backend;
    int n_threads;
    int ngl;
    bool warm_weights;
    qw3_think_mode default_think;
    float default_think_effort;
    bool default_think_effort_set;
    float default_temp;
    float default_top_p;
    int default_top_k;
    float default_min_p;
    float default_repeat_penalty;
    int default_repeat_last_n;
    int default_max_tokens;
    const char *api_key;
} srv_config;

typedef struct {
    int prompt_end;
    uint64_t turn_hash;
    qw3_tokens raw_tokens;
} cached_asst_turn;

typedef struct {
    uint64_t hash;
    size_t raw_len;
    float *embd;
    int n_tokens;
    int embd_dim;
    int grid_w;
    int grid_h;
    uint64_t last_use;
} srv_vis_cache_entry;

#define QW3_SRV_MAX_CACHED_TURNS 256

typedef struct {
    srv_config cfg;
    qw3_engine *engine;
    qw3_session *session;
#ifndef QW3_NO_METAL
    qw3_vision *vision;
#else
    void *vision;
#endif
    srv_vis_cache_entry vis_cache[QW3_SRV_VIS_CACHE_CAP];
    uint64_t vis_clock;
    pthread_mutex_t infer_lock;
    uint64_t req_counter;
    time_t started_at;
    /* Multi-turn KV prefix continuity across OpenAI assistant turns */
    cached_asst_turn cached_turns[QW3_SRV_MAX_CACHED_TURNS];
    int n_cached_turns;
    /* Inkling control token IDs cached once at startup */
    bool is_inkling;
    int ink_user;
    int ink_tool;
    int ink_model;
    int ink_text;
    int ink_image;
    int ink_code;
    int ink_think;
    int ink_call;
    int ink_end;
    int ink_eos;
} srv_state;

static uint64_t fnv1a_update(uint64_t h, const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void srv_clear_cached_turns(srv_state *srv) {
    for (int i = 0; i < srv->n_cached_turns; i++) {
        qw3_tokens_free(&srv->cached_turns[i].raw_tokens);
    }
    srv->n_cached_turns = 0;
}

static void srv_clear_vis_cache(srv_state *srv) {
    for (int i = 0; i < QW3_SRV_VIS_CACHE_CAP; i++) {
        free(srv->vis_cache[i].embd);
        memset(&srv->vis_cache[i], 0, sizeof(srv->vis_cache[i]));
    }
}

typedef struct {
    char id[64];
    char name[128];
    char *arguments;
} srv_tool_call;

typedef struct {
    srv_tool_call calls[QW3_SRV_MAX_TOOL_CALLS];
    int count;
} srv_tool_call_list;

static void srv_tool_calls_free(srv_tool_call_list *list) {
    for (int i = 0; i < list->count; i++) free(list->calls[i].arguments);
    list->count = 0;
}

/* =========================================================================
 * UTF-8 incomplete-sequence carry helper + GPT-2 byte-level BPE decoder
 * ========================================================================= */

static int srv_utf8_read_cp(const char *s, size_t len, size_t *pos, uint32_t *cp) {
    unsigned char c = (unsigned char)s[*pos];
    if (c < 0x80u) { *cp = c; (*pos)++; return 1; }
    if ((c & 0xE0u) == 0xC0u && *pos + 1 < len) {
        *cp = ((uint32_t)(c & 0x1Fu) << 6u) | (uint32_t)((unsigned char)s[*pos + 1] & 0x3Fu);
        *pos += 2; return 1;
    }
    if ((c & 0xF0u) == 0xE0u && *pos + 2 < len) {
        *cp = ((uint32_t)(c & 0x0Fu) << 12u) |
              ((uint32_t)((unsigned char)s[*pos + 1] & 0x3Fu) << 6u) |
              (uint32_t)((unsigned char)s[*pos + 2] & 0x3Fu);
        *pos += 3; return 1;
    }
    if ((c & 0xF8u) == 0xF0u && *pos + 3 < len) {
        *cp = ((uint32_t)(c & 0x07u) << 18u) |
              ((uint32_t)((unsigned char)s[*pos + 1] & 0x3Fu) << 12u) |
              ((uint32_t)((unsigned char)s[*pos + 2] & 0x3Fu) << 6u) |
              (uint32_t)((unsigned char)s[*pos + 3] & 0x3Fu);
        *pos += 4; return 1;
    }
    *cp = c; (*pos)++; return 0;
}

static int srv_gpt2_cp_to_byte(uint32_t cp, unsigned char *out) {
    if (((cp >= 33u && cp <= 126u) || (cp >= 161u && cp <= 172u) || cp >= 174u) && cp <= 255u) {
        *out = (unsigned char)cp;
        return 1;
    }
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256u; b++) {
        if ((b >= 33u && b <= 126u) || (b >= 161u && b <= 172u) || b >= 174u) continue;
        if (cp == 256u + n) { *out = (unsigned char)b; return 1; }
        n++;
    }
    return 0;
}

static char *srv_token_decoded_text(qw3_engine *engine, int token, size_t *out_len) {
    size_t raw_len = 0;
    char *raw = qw3_token_text(engine, token, &raw_len);
    if (!raw) { if (out_len) *out_len = 0; return NULL; }
    char *out = (char *)malloc(raw_len + 1);
    if (!out) { free(raw); if (out_len) *out_len = 0; return NULL; }
    size_t ip = 0, op = 0;
    while (ip < raw_len) {
        uint32_t cp = 0;
        size_t before = ip;
        srv_utf8_read_cp(raw, raw_len, &ip, &cp);
        unsigned char b = 0;
        if (srv_gpt2_cp_to_byte(cp, &b)) {
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

static size_t utf8_complete_prefix(const char *s, size_t len) {
    if (len == 0) return 0;
    size_t i = len;
    while (i > 0 && ((unsigned char)s[i - 1] & 0xC0u) == 0x80u) i--;
    if (i == 0) return len; /* malformed leading continuation bytes: flush */
    unsigned char lead = (unsigned char)s[i - 1];
    size_t expected = 1;
    if ((lead & 0x80u) == 0x00u) expected = 1;
    else if ((lead & 0xE0u) == 0xC0u) expected = 2;
    else if ((lead & 0xF0u) == 0xE0u) expected = 3;
    else if ((lead & 0xF8u) == 0xF0u) expected = 4;
    else return len;
    size_t avail = len - (i - 1);
    return avail < expected ? (i - 1) : len;
}

/* =========================================================================
 * Chat message rendering (Inkling-2-Mini and Qwen3)
 * ========================================================================= */

static bool extract_message_content(const jval *msg, strbuf *out, bool *saw_image) {
    const jval *c = jobj_get(msg, "content");
    if (!c || c->type == J_NULL) return true;
    if (c->type == J_STR) {
        sb_append_n(out, c->str, c->str_len);
        return true;
    }
    if (c->type == J_ARR) {
        for (size_t i = 0; i < c->len; i++) {
            const jval *part = c->items[i];
            if (!part || part->type != J_OBJ) continue;
            const char *ptype = jobj_str(part, "type");
            if (ptype && strcmp(ptype, "text") == 0) {
                const char *txt = jobj_str(part, "text");
                if (txt) sb_append(out, txt);
            } else if (ptype && (strcmp(ptype, "image_url") == 0 ||
                                 strcmp(ptype, "input_image") == 0)) {
                if (saw_image) *saw_image = true;
            }
        }
        return true;
    }
    return false;
}

static void build_inkling_tool_declare(const jval *tools, strbuf *out) {
    sb_append_c(out, '[');
    bool first = true;
    for (size_t i = 0; tools && tools->type == J_ARR && i < tools->len; i++) {
        const jval *t = tools->items[i];
        if (!t || t->type != J_OBJ) continue;
        const jval *fn = jobj_get(t, "function");
        const jval *src = (fn && fn->type == J_OBJ) ? fn : t;
        const char *name = jobj_str(src, "name");
        if (!name || !name[0]) continue;
        const char *desc = jobj_str(src, "description");
        const jval *params = jobj_get(src, "parameters");
        if (!first) sb_append_c(out, ',');
        first = false;
        sb_append(out, "{\"description\":");
        sb_append_json_str(out, desc ? desc : "");
        sb_append(out, ",\"name\":");
        sb_append_json_str(out, name);
        sb_append(out, ",\"parameters\":");
        if (params && params->raw && params->raw_len > 0) {
            sb_append_n(out, params->raw, params->raw_len);
        } else {
            sb_append(out, "{\"type\":\"object\",\"properties\":{}}");
        }
        sb_append(out, ",\"type\":\"function\"}");
    }
    sb_append_c(out, ']');
}

static void build_qwen_tools_system(const jval *tools, strbuf *sys_out) {
    if (!tools || tools->type != J_ARR || tools->len == 0) return;
    if (sys_out->len > 0) sb_append(sys_out, "\n\n");
    sb_append(sys_out,
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n");
    for (size_t i = 0; i < tools->len; i++) {
        const jval *t = tools->items[i];
        if (t && t->raw && t->raw_len > 0) {
            sb_append_n(sys_out, t->raw, t->raw_len);
            sb_append_c(sys_out, '\n');
        }
    }
    sb_append(sys_out,
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>");
}

/* Lookup tool name by tool_call_id from earlier assistant messages */
static const char *lookup_tool_name_by_id(const jval *messages, size_t upto, const char *tcid) {
    if (!messages || messages->type != J_ARR || !tcid || !tcid[0]) return NULL;
    for (size_t i = upto; i > 0; i--) {
        const jval *m = messages->items[i - 1];
        const char *role = jobj_str(m, "role");
        if (!role || strcmp(role, "assistant") != 0) continue;
        const jval *tcs = jobj_get(m, "tool_calls");
        if (!tcs || tcs->type != J_ARR) continue;
        for (size_t j = 0; j < tcs->len; j++) {
            const jval *tc = tcs->items[j];
            const char *id = jobj_str(tc, "id");
            if (id && strcmp(id, tcid) == 0) {
                const jval *fn = jobj_get(tc, "function");
                const char *nm = fn ? jobj_str(fn, "name") : jobj_str(tc, "name");
                if (nm && nm[0]) return nm;
            }
        }
    }
    return NULL;
}

static uint64_t hash_asst_turn_json(const strbuf *content, const jval *tcs) {
    uint64_t h = 14695981039346656037ull;
    if (content && content->p && content->len > 0)
        h = fnv1a_update(h, content->p, content->len);
    h = fnv1a_update(h, "\x01", 1);
    if (tcs && tcs->type == J_ARR) {
        for (size_t j = 0; j < tcs->len; j++) {
            const jval *tc = tcs->items[j];
            const jval *fn = jobj_get(tc, "function");
            const char *nm = fn ? jobj_str(fn, "name") : jobj_str(tc, "name");
            const jval *av = fn ? jobj_get(fn, "arguments") : jobj_get(tc, "arguments");
            if (nm) h = fnv1a_update(h, nm, strlen(nm));
            h = fnv1a_update(h, "\x02", 1);
            if (av && av->type == J_STR) h = fnv1a_update(h, av->str, av->str_len);
            else if (av && av->raw && av->raw_len > 0) h = fnv1a_update(h, av->raw, av->raw_len);
            h = fnv1a_update(h, "\x03", 1);
        }
    }
    return h;
}

static uint64_t hash_asst_turn_gen(const strbuf *content, const srv_tool_call_list *tcs) {
    uint64_t h = 14695981039346656037ull;
    if (content && content->p && content->len > 0)
        h = fnv1a_update(h, content->p, content->len);
    h = fnv1a_update(h, "\x01", 1);
    if (tcs) {
        for (int j = 0; j < tcs->count; j++) {
            h = fnv1a_update(h, tcs->calls[j].name, strlen(tcs->calls[j].name));
            h = fnv1a_update(h, "\x02", 1);
            if (tcs->calls[j].arguments)
                h = fnv1a_update(h, tcs->calls[j].arguments, strlen(tcs->calls[j].arguments));
            h = fnv1a_update(h, "\x03", 1);
        }
    }
    return h;
}

static bool try_splice_cached_asst(const srv_state *srv, qw3_tokens *out,
                                   const strbuf *content, const jval *tcs) {
    if (!srv->session || srv->n_cached_turns <= 0) return false;
    const qw3_tokens *sess_toks = qw3_session_tokens(srv->session);
    if (!sess_toks || out->len > sess_toks->len) return false;
    uint64_t h = hash_asst_turn_json(content, tcs);
    for (int i = 0; i < srv->n_cached_turns; i++) {
        const cached_asst_turn *ct = &srv->cached_turns[i];
        if (ct->prompt_end != out->len || ct->turn_hash != h) continue;
        if (out->len + ct->raw_tokens.len > sess_toks->len) continue;
        if (out->len > 0 && memcmp(out->v, sess_toks->v, (size_t)out->len * sizeof(int)) != 0)
            continue;
        if (ct->raw_tokens.len > 0 &&
            memcmp(ct->raw_tokens.v, sess_toks->v + out->len,
                   (size_t)ct->raw_tokens.len * sizeof(int)) != 0)
            continue;
        for (int k = 0; k < ct->raw_tokens.len; k++)
            qw3_tokens_push(out, ct->raw_tokens.v[k]);
        return true;
    }
    return false;
}

static void inkling_append_model_call(srv_state *srv, qw3_tokens *out,
                                      const char *fn_name, const char *args_json) {
    const char *nm = (fn_name && fn_name[0]) ? fn_name : "tool";
    const char *ag = (args_json && args_json[0]) ? args_json : "{}";
    strbuf wrapped;
    sb_init(&wrapped);
    sb_append(&wrapped, "{\"name\":");
    sb_append_json_str(&wrapped, nm);
    sb_append(&wrapped, ",\"args\":");
    sb_append(&wrapped, ag);
    sb_append_c(&wrapped, '}');
    qw3_tokens_push(out, srv->ink_model);
    qw3_tokenize_text(srv->engine, nm, out);
    qw3_tokens_push(out, srv->ink_call);
    qw3_tokenize_text(srv->engine, wrapped.p, out);
    qw3_tokens_push(out, srv->ink_end);
    sb_free(&wrapped);
}

/* =========================================================================
 * Multimodal prompt plan, base64/file image decoding & vision cache
 * ========================================================================= */

typedef struct {
    bool is_image;
    int start_pos;      /* index in plan->tokens */
    int len;            /* number of tokens / visual embeddings */
    const float *embd;  /* borrowed from srv->vis_cache when is_image */
    int embd_dim;
} srv_prompt_seg;

typedef struct {
    qw3_tokens tokens;  /* text token IDs (>=0) + negative image sentinels */
    srv_prompt_seg *segs;
    int n_segs;
    int cap_segs;
    bool has_images;
    int n_images;
    double vis_encode_ms;
} srv_prompt_plan;

static void srv_prompt_plan_free(srv_prompt_plan *pl) {
    if (!pl) return;
    qw3_tokens_free(&pl->tokens);
    free(pl->segs);
    memset(pl, 0, sizeof(*pl));
}

static bool srv_plan_push_seg(srv_prompt_plan *pl, srv_prompt_seg seg) {
    if (pl->n_segs >= pl->cap_segs) {
        int ncap = pl->cap_segs ? pl->cap_segs * 2 : 8;
        srv_prompt_seg *ns = (srv_prompt_seg *)realloc(pl->segs, (size_t)ncap * sizeof(*ns));
        if (!ns) return false;
        pl->segs = ns;
        pl->cap_segs = ncap;
    }
    pl->segs[pl->n_segs++] = seg;
    return true;
}

static bool srv_plan_flush_text(srv_prompt_plan *pl, int text_start) {
    int len = pl->tokens.len - text_start;
    if (len <= 0) return true;
    srv_prompt_seg seg;
    memset(&seg, 0, sizeof(seg));
    seg.is_image = false;
    seg.start_pos = text_start;
    seg.len = len;
    return srv_plan_push_seg(pl, seg);
}

#ifndef QW3_NO_METAL
static int vis_sentinel_id(uint64_t img_hash, int idx) {
    uint64_t h = img_hash ^ ((uint64_t)(idx + 1) * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return -1000 - (int)(h & 0x3FFFFFFFu);
}

static int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

static uint8_t *decode_base64(const char *s, size_t len, size_t *out_len) {
    uint8_t *out = (uint8_t *)malloc((len * 3) / 4 + 4);
    if (!out) return NULL;
    size_t op = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '=') break;
        if (isspace(c)) continue;
        int v = b64_val(c);
        if (v < 0) { free(out); return NULL; }
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[op++] = (uint8_t)((acc >> bits) & 0xFFu);
        }
    }
    *out_len = op;
    return out;
}

static uint8_t *load_image_url_bytes(const char *url, size_t *out_len, char *err, size_t errlen) {
    if (!url || !url[0]) {
        snprintf(err, errlen, "empty image_url");
        return NULL;
    }
    if (strncmp(url, "data:", 5) == 0) {
        const char *comma = strchr(url, ',');
        if (!comma) {
            snprintf(err, errlen, "malformed data: URI in image_url");
            return NULL;
        }
        bool is_b64 = false;
        for (const char *p = url + 5; p + 7 <= comma; p++) {
            if (strncasecmp(p, ";base64", 7) == 0) { is_b64 = true; break; }
        }
        if (!is_b64) {
            snprintf(err, errlen, "data: image_url must be base64-encoded");
            return NULL;
        }
        const char *payload = comma + 1;
        uint8_t *bytes = decode_base64(payload, strlen(payload), out_len);
        if (!bytes || *out_len == 0) {
            free(bytes);
            snprintf(err, errlen, "invalid base64 payload in image_url");
            return NULL;
        }
        return bytes;
    }
    const char *path = NULL;
    if (strncmp(url, "file://", 7) == 0) {
        path = url + 7;
    } else if (url[0] == '/') {
        path = url;
    } else {
        snprintf(err, errlen, "unsupported image_url scheme (use data:image/...;base64,... or file://...)");
        return NULL;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, errlen, "cannot open image file '%s': %s", path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long sz = ftell(fp);
    if (sz <= 0 || sz > 64L * 1024L * 1024L) {
        fclose(fp);
        snprintf(err, errlen, "invalid image file size (%ld bytes)", sz);
        return NULL;
    }
    rewind(fp);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(fp); return NULL; }
    size_t nr = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (nr != (size_t)sz) {
        free(buf);
        snprintf(err, errlen, "short read on image file '%s'", path);
        return NULL;
    }
    *out_len = (size_t)sz;
    return buf;
}
#endif

static const char *extract_part_image_url(const jval *part) {
    const jval *iu = jobj_get(part, "image_url");
    if (iu) {
        if (iu->type == J_STR) return iu->str;
        if (iu->type == J_OBJ) return jobj_str(iu, "url");
    }
    const char *u = jobj_str(part, "url");
    if (u) return u;
    return jobj_str(part, "image");
}

static bool srv_append_image_part(srv_state *srv, srv_prompt_plan *plan,
                                  const char *url, char *err, size_t errlen) {
#ifdef QW3_NO_METAL
    (void)srv; (void)plan; (void)url;
    snprintf(err, errlen, "image_url input requires vision support (--mmproj), not available in CPU-only build");
    return false;
#else
    if (!srv->vision) {
        snprintf(err, errlen, "image_url input requires vision support (--mmproj), not yet loaded");
        return false;
    }
    if (!srv->is_inkling) {
        snprintf(err, errlen, "loaded vision mmproj currently supports Inkling-2-Mini models");
        return false;
    }
    size_t raw_len = 0;
    uint8_t *raw = load_image_url_bytes(url, &raw_len, err, errlen);
    if (!raw) return false;

    uint64_t img_hash = fnv1a_update(14695981039346656037ull, raw, raw_len);

    /* Lookup in LRU vision embedding cache */
    srv_vis_cache_entry *entry = NULL;
    for (int i = 0; i < QW3_SRV_VIS_CACHE_CAP; i++) {
        if (srv->vis_cache[i].embd &&
            srv->vis_cache[i].hash == img_hash &&
            srv->vis_cache[i].raw_len == raw_len) {
            entry = &srv->vis_cache[i];
            entry->last_use = ++srv->vis_clock;
            break;
        }
    }

    if (!entry) {
        qw3_vision_result vr;
        if (qw3_vision_encode_bytes(srv->vision, raw, raw_len, &vr, err, errlen) != 0) {
            free(raw);
            return false;
        }
        plan->vis_encode_ms += vr.encode_ms;
        /* Pick empty or LRU slot */
        int slot = 0;
        for (int i = 0; i < QW3_SRV_VIS_CACHE_CAP; i++) {
            if (!srv->vis_cache[i].embd) { slot = i; break; }
            if (srv->vis_cache[i].last_use < srv->vis_cache[slot].last_use) slot = i;
        }
        free(srv->vis_cache[slot].embd);
        entry = &srv->vis_cache[slot];
        entry->hash = img_hash;
        entry->raw_len = raw_len;
        entry->embd = vr.embd; /* transfer ownership */
        entry->n_tokens = vr.n_tokens;
        entry->embd_dim = vr.embd_dim;
        entry->grid_w = vr.grid_w;
        entry->grid_h = vr.grid_h;
        entry->last_use = ++srv->vis_clock;
    }
    free(raw);

    srv_prompt_seg seg;
    memset(&seg, 0, sizeof(seg));
    seg.is_image = true;
    seg.start_pos = plan->tokens.len;
    seg.len = entry->n_tokens;
    seg.embd = entry->embd;
    seg.embd_dim = entry->embd_dim;
    if (!srv_plan_push_seg(plan, seg)) {
        snprintf(err, errlen, "out of memory appending image segment");
        return false;
    }
    for (int k = 0; k < entry->n_tokens; k++) {
        qw3_tokens_push(&plan->tokens, vis_sentinel_id(img_hash, k));
    }
    plan->has_images = true;
    plan->n_images++;
    return true;
#endif
}

static bool message_has_image_part(const jval *msg) {
    const jval *c = jobj_get(msg, "content");
    if (!c || c->type != J_ARR) return false;
    for (size_t i = 0; i < c->len; i++) {
        const jval *part = c->items[i];
        if (!part || part->type != J_OBJ) continue;
        const char *ptype = jobj_str(part, "type");
        if (ptype && (strcmp(ptype, "image_url") == 0 ||
                      strcmp(ptype, "input_image") == 0 ||
                      strcmp(ptype, "image") == 0)) {
            return true;
        }
    }
    return false;
}

static bool encode_openai_chat_plan(srv_state *srv, const jval *messages, const jval *tools,
                                    qw3_think_mode think_mode,
                                    bool think_effort_set, float think_effort,
                                    srv_prompt_plan *plan,
                                    char *err, size_t errlen) {
    memset(plan, 0, sizeof(*plan));
    qw3_tokens *out = &plan->tokens;
    int text_start = 0;

    if (!messages || messages->type != J_ARR || messages->len == 0) {
        snprintf(err, errlen, "messages must be a non-empty array");
        return false;
    }
    bool saw_image = false;
    if (srv->is_inkling) {
        if (tools && tools->type == J_ARR && tools->len > 0) {
            strbuf tdecl;
            sb_init(&tdecl);
            build_inkling_tool_declare(tools, &tdecl);
            if (tdecl.len > 2) {
                qw3_chat_append_message(srv->engine, out, "tool_declare", tdecl.p);
            }
            sb_free(&tdecl);
        }
        /* Combine any leading system/developer messages, then append effort */
        strbuf sys;
        sb_init(&sys);
        size_t idx = 0;
        while (idx < messages->len) {
            const jval *m = messages->items[idx];
            const char *role = jobj_str(m, "role");
            if (!role || (strcmp(role, "system") != 0 && strcmp(role, "developer") != 0)) break;
            if (sys.len > 0) sb_append_c(&sys, '\n');
            extract_message_content(m, &sys, &saw_image);
            idx++;
        }
        if (saw_image) {
            sb_free(&sys);
            snprintf(err, errlen, "image content is only supported for user and tool messages");
            return false;
        }
        if (sys.len > 0) {
            qw3_chat_append_message(srv->engine, out, "system", sys.p);
        }
        sb_free(&sys);
        if (think_effort_set)
            qw3_inkling_chat_append_effort_level(srv->engine, out, think_effort);
        else
            qw3_inkling_chat_append_effort(srv->engine, out, think_mode);

        for (size_t i = idx; i < messages->len; i++) {
            const jval *m = messages->items[i];
            if (!m || m->type != J_OBJ) continue;
            const char *role = jobj_str(m, "role");
            if (!role) role = "user";

            /* Multimodal array content for user/tool messages (exact Inkling chat_template.jinja) */
            if ((strcmp(role, "user") == 0 || strcmp(role, "tool") == 0) && message_has_image_part(m)) {
                const jval *carr = jobj_get(m, "content");
                const char *tname = NULL;
                if (strcmp(role, "tool") == 0) {
                    tname = jobj_str(m, "name");
                    if (!tname || !tname[0]) {
                        tname = lookup_tool_name_by_id(messages, i, jobj_str(m, "tool_call_id"));
                    }
                    if (!tname || !tname[0]) tname = "tool";
                }
                for (size_t pi = 0; pi < carr->len; pi++) {
                    const jval *part = carr->items[pi];
                    if (!part) continue;
                    if (part->type == J_STR) {
                        qw3_tokens_push(out, strcmp(role, "tool") == 0 ? srv->ink_tool : srv->ink_user);
                        if (tname) qw3_tokenize_text(srv->engine, tname, out);
                        qw3_tokens_push(out, srv->ink_text);
                        qw3_tokenize_text(srv->engine, part->str, out);
                        qw3_tokens_push(out, srv->ink_end);
                        continue;
                    }
                    if (part->type != J_OBJ) continue;
                    const char *ptype = jobj_str(part, "type");
                    if (!ptype || strcmp(ptype, "text") == 0 || strcmp(ptype, "input_text") == 0) {
                        const char *txt = jobj_str(part, "text");
                        qw3_tokens_push(out, strcmp(role, "tool") == 0 ? srv->ink_tool : srv->ink_user);
                        if (tname) qw3_tokenize_text(srv->engine, tname, out);
                        qw3_tokens_push(out, srv->ink_text);
                        if (txt && txt[0]) qw3_tokenize_text(srv->engine, txt, out);
                        qw3_tokens_push(out, srv->ink_end);
                    } else if (strcmp(ptype, "image_url") == 0 ||
                               strcmp(ptype, "input_image") == 0 ||
                               strcmp(ptype, "image") == 0) {
                        const char *url = extract_part_image_url(part);
                        qw3_tokens_push(out, strcmp(role, "tool") == 0 ? srv->ink_tool : srv->ink_user);
                        if (tname) qw3_tokenize_text(srv->engine, tname, out);
                        qw3_tokens_push(out, srv->ink_image);
                        if (!srv_plan_flush_text(plan, text_start)) {
                            snprintf(err, errlen, "out of memory flushing text segment");
                            return false;
                        }
                        if (!srv_append_image_part(srv, plan, url, err, errlen)) {
                            return false;
                        }
                        text_start = out->len;
                        qw3_tokens_push(out, srv->ink_end);
                    }
                }
                continue;
            }

            strbuf content;
            sb_init(&content);
            extract_message_content(m, &content, &saw_image);
            if (saw_image) {
                sb_free(&content);
                snprintf(err, errlen, "image content is only supported for user and tool messages");
                return false;
            }
            if (strcmp(role, "system") == 0 || strcmp(role, "developer") == 0) {
                qw3_chat_append_message(srv->engine, out, "system", content.p ? content.p : "");
            } else if (strcmp(role, "user") == 0) {
                qw3_chat_append_message(srv->engine, out, "user", content.p ? content.p : "");
            } else if (strcmp(role, "tool") == 0) {
                const char *tname = jobj_str(m, "name");
                if (!tname || !tname[0]) {
                    tname = lookup_tool_name_by_id(messages, i, jobj_str(m, "tool_call_id"));
                }
                char role_buf[160];
                snprintf(role_buf, sizeof(role_buf), "tool:%s", (tname && tname[0]) ? tname : "tool");
                qw3_chat_append_message(srv->engine, out, role_buf, content.p ? content.p : "");
            } else if (strcmp(role, "assistant") == 0) {
                const jval *tcs = jobj_get(m, "tool_calls");
                if (try_splice_cached_asst(srv, out, &content, tcs)) {
                    /* Exact cached session tokens spliced for 100% KV prefix reuse */
                } else if (tcs && tcs->type == J_ARR && tcs->len > 0) {
                    if (content.len > 0) {
                        qw3_tokens_push(out, srv->ink_model);
                        qw3_tokens_push(out, srv->ink_text);
                        qw3_tokenize_text(srv->engine, content.p, out);
                        qw3_tokens_push(out, srv->ink_end);
                    }
                    for (size_t j = 0; j < tcs->len; j++) {
                        const jval *tc = tcs->items[j];
                        const jval *fn = jobj_get(tc, "function");
                        const char *nm = fn ? jobj_str(fn, "name") : jobj_str(tc, "name");
                        const jval *av = fn ? jobj_get(fn, "arguments") : jobj_get(tc, "arguments");
                        const char *args = "{}";
                        strbuf raw_args;
                        sb_init(&raw_args);
                        if (av && av->type == J_STR) args = av->str;
                        else if (av && av->raw && av->raw_len > 0) {
                            sb_append_n(&raw_args, av->raw, av->raw_len);
                            args = raw_args.p;
                        }
                        inkling_append_model_call(srv, out, nm ? nm : "tool", args);
                        sb_free(&raw_args);
                    }
                    qw3_tokens_push(out, srv->ink_eos);
                } else {
                    qw3_chat_append_message(srv->engine, out, "assistant", content.p ? content.p : "");
                }
            }
            sb_free(&content);
        }
        return srv_plan_flush_text(plan, text_start);
    }

    /* Qwen3 ChatML encoding */
    strbuf sys;
    sb_init(&sys);
    size_t idx = 0;
    while (idx < messages->len) {
        const jval *m = messages->items[idx];
        const char *role = jobj_str(m, "role");
        if (!role || (strcmp(role, "system") != 0 && strcmp(role, "developer") != 0)) break;
        if (sys.len > 0) sb_append_c(&sys, '\n');
        extract_message_content(m, &sys, &saw_image);
        idx++;
    }
    if (saw_image) {
        sb_free(&sys);
        snprintf(err, errlen, "image_url input requires vision support (--mmproj), not yet loaded");
        return false;
    }
    build_qwen_tools_system(tools, &sys);
    if (sys.len > 0) {
        qw3_chat_append_message(srv->engine, out, "system", sys.p);
    }
    sb_free(&sys);

    for (size_t i = idx; i < messages->len; i++) {
        const jval *m = messages->items[i];
        if (!m || m->type != J_OBJ) continue;
        const char *role = jobj_str(m, "role");
        if (!role) role = "user";
        strbuf content;
        sb_init(&content);
        extract_message_content(m, &content, &saw_image);
        if (saw_image) {
            sb_free(&content);
            snprintf(err, errlen, "image_url input requires vision support (--mmproj), not yet loaded");
            return false;
        }
        if (strcmp(role, "tool") == 0) {
            strbuf wrapped;
            sb_init(&wrapped);
            sb_append(&wrapped, "<tool_response>\n");
            if (content.len > 0) sb_append_n(&wrapped, content.p, content.len);
            sb_append(&wrapped, "\n</tool_response>");
            qw3_chat_append_message(srv->engine, out, "user", wrapped.p);
            sb_free(&wrapped);
        } else if (strcmp(role, "assistant") == 0) {
            const jval *tcs = jobj_get(m, "tool_calls");
            if (try_splice_cached_asst(srv, out, &content, tcs)) {
                /* Exact cached session tokens spliced for 100% KV prefix reuse */
            } else {
                if (tcs && tcs->type == J_ARR && tcs->len > 0) {
                    for (size_t j = 0; j < tcs->len; j++) {
                        const jval *tc = tcs->items[j];
                        const jval *fn = jobj_get(tc, "function");
                        const char *nm = fn ? jobj_str(fn, "name") : jobj_str(tc, "name");
                        const jval *av = fn ? jobj_get(fn, "arguments") : jobj_get(tc, "arguments");
                        if (content.len > 0) sb_append_c(&content, '\n');
                        sb_append(&content, "<tool_call>\n{\"name\":");
                        sb_append_json_str(&content, nm ? nm : "tool");
                        sb_append(&content, ",\"arguments\":");
                        if (av && av->type == J_STR) sb_append(&content, av->str);
                        else if (av && av->raw && av->raw_len > 0) sb_append_n(&content, av->raw, av->raw_len);
                        else sb_append(&content, "{}");
                        sb_append(&content, "}\n</tool_call>");
                    }
                }
                qw3_chat_append_message(srv->engine, out, "assistant", content.p ? content.p : "");
            }
        } else {
            qw3_chat_append_message(srv->engine, out,
                strcmp(role, "developer") == 0 ? "system" : role,
                content.p ? content.p : "");
        }
        sb_free(&content);
    }
    qw3_chat_append_assistant_prefix(srv->engine, out, think_mode);
    return srv_plan_flush_text(plan, text_start);
}

static int srv_sync_prompt_plan(srv_state *srv, const srv_prompt_plan *plan, int fd,
                                int *out_cached, char *err, size_t errlen) {
    int session_pos = qw3_session_pos(srv->session);
    int common = qw3_session_common_prefix(srv->session, &plan->tokens);
    int cached = (common == session_pos && plan->tokens.len >= common) ? common : 0;
    if (out_cached) *out_cached = cached;
    if (cached == 0) {
        srv_clear_cached_turns(srv);
    }

    if (!plan->has_images) {
        return qw3_session_sync_interruptible(srv->session, &plan->tokens,
                                              http_interrupt_cb, &fd,
                                              err, errlen);
    }

    if (cached != session_pos) {
        qw3_session_invalidate(srv->session);
        cached = 0;
        if (out_cached) *out_cached = 0;
    }

    for (int i = 0; i < plan->n_segs; i++) {
        const srv_prompt_seg *seg = &plan->segs[i];
        int seg_end = seg->start_pos + seg->len;
        if (seg_end <= cached) continue;
        int skip = (cached > seg->start_pos) ? (cached - seg->start_pos) : 0;
        int rem = seg->len - skip;
        if (rem <= 0) continue;
        bool is_last = (seg_end == plan->tokens.len);
        int rc = 0;
        if (!seg->is_image) {
            rc = qw3_session_prefill_tokens_interruptible(
                srv->session, plan->tokens.v + seg->start_pos + skip, rem,
                is_last, http_interrupt_cb, &fd, err, errlen);
        } else {
            rc = qw3_session_eval_embeddings_interruptible(
                srv->session,
                seg->embd + (size_t)skip * (size_t)seg->embd_dim,
                rem, seg->embd_dim,
                plan->tokens.v + seg->start_pos + skip,
                is_last, http_interrupt_cb, &fd, err, errlen);
        }
        if (rc != 0) return rc;
    }
    return 0;
}

/* =========================================================================
 * Qwen3 <tool_call> post-extractor (JSON or XML format)
 * ========================================================================= */

static void parse_qwen_tool_calls(const char *text, srv_tool_call_list *calls, strbuf *clean_out) {
    if (!text) return;
    const char *p = text;
    while (*p) {
        const char *open = strstr(p, "<tool_call>");
        if (!open) {
            sb_append(clean_out, p);
            break;
        }
        sb_append_n(clean_out, p, (size_t)(open - p));
        const char *body = open + 11;
        const char *close = strstr(body, "</tool_call>");
        if (!close) {
            sb_append(clean_out, open);
            break;
        }
        while (body < close && (*body == ' ' || *body == '\t' || *body == '\r' || *body == '\n')) body++;
        const char *bend = close;
        while (bend > body && (bend[-1] == ' ' || bend[-1] == '\t' || bend[-1] == '\r' || bend[-1] == '\n')) bend--;

        if (calls->count < QW3_SRV_MAX_TOOL_CALLS && bend > body) {
            jval *jv = json_parse(body, (size_t)(bend - body));
            if (jv && jv->type == J_OBJ) {
                const char *nm = jobj_str(jv, "name");
                const jval *args = jobj_get(jv, "arguments");
                if (nm && nm[0]) {
                    srv_tool_call *tc = &calls->calls[calls->count];
                    snprintf(tc->id, sizeof(tc->id), "call_%d", calls->count);
                    snprintf(tc->name, sizeof(tc->name), "%s", nm);
                    strbuf ab;
                    sb_init(&ab);
                    if (args && args->type == J_STR) sb_append_n(&ab, args->str, args->str_len);
                    else if (args && args->raw && args->raw_len > 0) sb_append_n(&ab, args->raw, args->raw_len);
                    else sb_append(&ab, "{}");
                    tc->arguments = ab.p;
                    calls->count++;
                }
            }
            jval_free(jv);
        }
        p = close + 12;
    }
}

/* =========================================================================
 * SSE chunk builders
 * ========================================================================= */

static bool sse_send_chat_delta(int fd, const char *req_id, time_t created, const char *model,
                                const char *role, const char *content, size_t content_len,
                                const char *reasoning, size_t reasoning_len,
                                const srv_tool_call *tc, int tc_index,
                                const char *finish_reason) {
    strbuf sb;
    sb_init(&sb);
    sb_append(&sb, "{\"id\":");
    sb_append_json_str(&sb, req_id);
    sb_appendf(&sb, ",\"object\":\"chat.completion.chunk\",\"created\":%lld,\"model\":",
               (long long)created);
    sb_append_json_str(&sb, model);
    sb_append(&sb, ",\"choices\":[{\"index\":0,\"delta\":{");
    bool need_comma = false;
    if (role) {
        sb_append(&sb, "\"role\":");
        sb_append_json_str(&sb, role);
        need_comma = true;
    }
    if (reasoning && reasoning_len > 0) {
        if (need_comma) sb_append_c(&sb, ',');
        sb_append(&sb, "\"reasoning_content\":");
        sb_append_json_str_n(&sb, reasoning, reasoning_len);
        need_comma = true;
    }
    if (content && content_len > 0) {
        if (need_comma) sb_append_c(&sb, ',');
        sb_append(&sb, "\"content\":");
        sb_append_json_str_n(&sb, content, content_len);
        need_comma = true;
    }
    if (tc) {
        if (need_comma) sb_append_c(&sb, ',');
        sb_appendf(&sb, "\"tool_calls\":[{\"index\":%d,\"id\":", tc_index);
        sb_append_json_str(&sb, tc->id);
        sb_append(&sb, ",\"type\":\"function\",\"function\":{\"name\":");
        sb_append_json_str(&sb, tc->name);
        sb_append(&sb, ",\"arguments\":");
        sb_append_json_str(&sb, tc->arguments ? tc->arguments : "{}");
        sb_append(&sb, "}}]");
    }
    sb_append(&sb, "},\"finish_reason\":");
    if (finish_reason) sb_append_json_str(&sb, finish_reason);
    else sb_append(&sb, "null");
    sb_append(&sb, "}]}");
    bool ok = send_sse_data(fd, sb.p, sb.len);
    sb_free(&sb);
    return ok;
}

static bool sse_send_usage_chunk(int fd, const char *req_id, time_t created, const char *model,
                                 int prompt_tokens, int cached_tokens, int completion_tokens,
                                 double prefill_ms, int prefill_n, double gen_ms) {
    strbuf sb;
    sb_init(&sb);
    sb_append(&sb, "{\"id\":");
    sb_append_json_str(&sb, req_id);
    sb_appendf(&sb, ",\"object\":\"chat.completion.chunk\",\"created\":%lld,\"model\":",
               (long long)created);
    sb_append_json_str(&sb, model);
    sb_appendf(&sb,
        ",\"choices\":[],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
        "\"prompt_tokens_details\":{\"cached_tokens\":%d}},"
        "\"timings\":{\"cache_n\":%d,\"prompt_n\":%d,\"prompt_ms\":%.2f,\"prompt_per_second\":%.2f,"
        "\"predicted_n\":%d,\"predicted_ms\":%.2f,\"predicted_per_second\":%.2f}}",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, cached_tokens,
        cached_tokens, prefill_n, prefill_ms, prefill_ms > 0.0 ? (1000.0 * prefill_n / prefill_ms) : 0.0,
        completion_tokens, gen_ms, gen_ms > 0.0 ? (1000.0 * completion_tokens / gen_ms) : 0.0);
    bool ok = send_sse_data(fd, sb.p, sb.len);
    sb_free(&sb);
    return ok;
}

/* =========================================================================
 * Sampling / request parameter extraction
 * ========================================================================= */

typedef struct {
    float temperature;
    float top_p;
    int top_k;
    float min_p;
    float repeat_penalty;
    int repeat_last_n;
    int max_tokens;
    uint64_t seed;
    bool stream;
    bool include_usage;
    bool has_tools;
    qw3_think_mode think_mode;
    float think_effort;
    bool think_effort_set;
    char *stops[QW3_SRV_MAX_STOPS];
    size_t stop_lens[QW3_SRV_MAX_STOPS];
    int n_stops;
    size_t max_stop_len;
} req_params;

static void req_params_free(req_params *p) {
    for (int i = 0; i < p->n_stops; i++) free(p->stops[i]);
    p->n_stops = 0;
}

static void parse_req_params(const srv_config *cfg, const jval *body, req_params *p) {
    memset(p, 0, sizeof(*p));
    p->temperature = (float)jobj_num(body, "temperature", cfg->default_temp);
    p->top_p = (float)jobj_num(body, "top_p", cfg->default_top_p);
    p->top_k = (int)jobj_num(body, "top_k", cfg->default_top_k);
    p->min_p = (float)jobj_num(body, "min_p", cfg->default_min_p);
    p->repeat_penalty = (float)jobj_num(body, "repeat_penalty", cfg->default_repeat_penalty);
    p->repeat_last_n = (int)jobj_num(body, "repeat_last_n", cfg->default_repeat_last_n);
    double mt = jobj_num(body, "max_completion_tokens",
                         jobj_num(body, "max_tokens", cfg->default_max_tokens));
    p->max_tokens = mt > 0 ? (int)mt : cfg->default_max_tokens;
    const jval *sv = jobj_get(body, "seed");
    if (sv && sv->type == J_NUM) {
        p->seed = (uint64_t)(int64_t)sv->num ^ 0x9e3779b97f4a7c15ull;
    } else {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        p->seed = ((uint64_t)ts.tv_nsec << 16u) ^ (uint64_t)ts.tv_sec ^ 0x123456789abcdef0ull;
    }
    if (p->seed == 0) p->seed = 1;
    p->stream = jobj_bool(body, "stream", false);
    const jval *so = jobj_get(body, "stream_options");
    p->include_usage = (so && so->type == J_OBJ) ? jobj_bool(so, "include_usage", false) : false;

    const jval *tools = jobj_get(body, "tools");
    p->has_tools = (tools && tools->type == J_ARR && tools->len > 0);

    /* Thinking mode resolution: reasoning_effort / think_effort / enable_thinking / chat_template_kwargs */
    p->think_mode = cfg->default_think;
    p->think_effort = cfg->default_think_effort;
    p->think_effort_set = cfg->default_think_effort_set;
    const jval *re_val = jobj_get(body, "reasoning_effort");
    if (re_val) {
        if (re_val->type == J_NUM && isfinite(re_val->num)) {
            float ef = (float)re_val->num;
            if (ef < 0.0f) ef = 0.0f;
            if (ef > 0.99f) ef = 0.99f;
            p->think_effort = ef;
            p->think_effort_set = true;
            p->think_mode = (ef <= 0.0f) ? QW3_THINK_NONE : QW3_THINK_ON;
        } else if (re_val->type == J_STR && re_val->str) {
            const char *re = re_val->str;
            if (strcmp(re, "none") == 0 || strcmp(re, "off") == 0 || strcmp(re, "0") == 0) {
                p->think_mode = QW3_THINK_NONE;
                p->think_effort = 0.0f;
                p->think_effort_set = true;
            } else if (strcmp(re, "low") == 0) {
                p->think_mode = QW3_THINK_ON;
                p->think_effort = 0.3f;
                p->think_effort_set = true;
            } else if (strcmp(re, "medium") == 0) {
                p->think_mode = QW3_THINK_ON;
                p->think_effort = 0.6f;
                p->think_effort_set = true;
            } else if (strcmp(re, "high") == 0) {
                p->think_mode = QW3_THINK_HIGH;
                p->think_effort = 0.9f;
                p->think_effort_set = true;
            } else if (strcmp(re, "max") == 0) {
                p->think_mode = QW3_THINK_HIGH;
                p->think_effort = 0.99f;
                p->think_effort_set = true;
            } else {
                p->think_mode = QW3_THINK_ON;
            }
        }
    }
    const jval *te = jobj_get(body, "think_effort");
    if (te && te->type == J_NUM && isfinite(te->num)) {
        float ef = (float)te->num;
        if (ef < 0.0f) ef = 0.0f;
        if (ef > 0.99f) ef = 0.99f;
        p->think_effort = ef;
        p->think_effort_set = true;
        p->think_mode = (ef <= 0.0f) ? QW3_THINK_NONE : QW3_THINK_ON;
    }
    const jval *et = jobj_get(body, "enable_thinking");
    if (et && et->type == J_BOOL) {
        p->think_mode = et->b ? QW3_THINK_ON : QW3_THINK_NONE;
        if (!et->b) {
            p->think_effort = 0.0f;
            p->think_effort_set = true;
        } else if (p->think_effort_set && p->think_effort <= 0.0f) {
            p->think_effort_set = false;
        }
    }
    const jval *ctk = jobj_get(body, "chat_template_kwargs");
    if (ctk && ctk->type == J_OBJ) {
        const jval *cet = jobj_get(ctk, "enable_thinking");
        if (cet && cet->type == J_BOOL) {
            p->think_mode = cet->b ? QW3_THINK_ON : QW3_THINK_NONE;
            if (!cet->b) {
                p->think_effort = 0.0f;
                p->think_effort_set = true;
            } else if (p->think_effort_set && p->think_effort <= 0.0f) {
                p->think_effort_set = false;
            }
        }
    }

    /* Stop sequences */
    const jval *stop = jobj_get(body, "stop");
    if (stop && stop->type == J_STR && stop->str_len > 0) {
        p->stops[0] = strdup(stop->str);
        p->stop_lens[0] = stop->str_len;
        p->n_stops = 1;
        p->max_stop_len = stop->str_len;
    } else if (stop && stop->type == J_ARR) {
        for (size_t i = 0; i < stop->len && p->n_stops < QW3_SRV_MAX_STOPS; i++) {
            const jval *it = stop->items[i];
            if (it && it->type == J_STR && it->str_len > 0) {
                p->stops[p->n_stops] = strdup(it->str);
                p->stop_lens[p->n_stops] = it->str_len;
                if (it->str_len > p->max_stop_len) p->max_stop_len = it->str_len;
                p->n_stops++;
            }
        }
    }
}

/* Returns byte offset in buf where earliest stop sequence begins, or SIZE_MAX */
static size_t find_stop_seq(const strbuf *buf, const req_params *p) {
    size_t best = SIZE_MAX;
    if (!buf->p || buf->len == 0) return best;
    for (int i = 0; i < p->n_stops; i++) {
        if (!p->stops[i] || p->stop_lens[i] == 0) continue;
        const char *hit = strstr(buf->p, p->stops[i]);
        if (hit) {
            size_t off = (size_t)(hit - buf->p);
            if (off < best) best = off;
        }
    }
    return best;
}

static int sample_next_token(qw3_session *s, const req_params *p, uint64_t *rng,
                             const qw3_tokens *recent) {
    if (p->temperature <= 0.0f && p->repeat_penalty <= 1.0f) {
        return qw3_session_argmax(s);
    }
    int n_rec = recent ? recent->len : 0;
    const int *rec_ptr = recent ? recent->v : NULL;
    if (p->repeat_last_n > 0 && n_rec > p->repeat_last_n) {
        rec_ptr += (n_rec - p->repeat_last_n);
        n_rec = p->repeat_last_n;
    }
    if (p->repeat_penalty > 1.0f && n_rec > 0) {
        return qw3_session_sample_repetition(s, p->temperature, p->top_k,
                                             p->top_p, p->min_p, rng,
                                             rec_ptr, n_rec, p->repeat_penalty);
    }
    return qw3_session_sample(s, p->temperature, p->top_k, p->top_p, p->min_p, rng);
}

/* =========================================================================
 * /v1/chat/completions handler
 * ========================================================================= */

static void handle_chat_completions(srv_state *srv, int fd, const jval *body) {
    const jval *messages = jobj_get(body, "messages");
    const jval *tools = jobj_get(body, "tools");
    req_params p;
    parse_req_params(&srv->cfg, body, &p);

    char err[256] = {0};
    srv_prompt_plan plan;
    memset(&plan, 0, sizeof(plan));
    pthread_mutex_lock(&srv->infer_lock);
    if (!encode_openai_chat_plan(srv, messages, tools, p.think_mode,
                                 p.think_effort_set, p.think_effort,
                                 &plan, err, sizeof(err))) {
        pthread_mutex_unlock(&srv->infer_lock);
        send_http_error(fd, 400, "Bad Request", "invalid_request_error", err);
        srv_prompt_plan_free(&plan);
        req_params_free(&p);
        return;
    }
    const qw3_tokens *prompt = &plan.tokens;
    if (prompt->len == 0 || prompt->len >= srv->cfg.ctx_size) {
        pthread_mutex_unlock(&srv->infer_lock);
        snprintf(err, sizeof(err), "prompt length (%d tokens) exceeds context size (%d)",
                 prompt->len, srv->cfg.ctx_size);
        send_http_error(fd, 400, "Bad Request", "invalid_request_error", err);
        srv_prompt_plan_free(&plan);
        req_params_free(&p);
        return;
    }

    uint64_t rseq = ++srv->req_counter;
    time_t created = time(NULL);
    char req_id[64];
    snprintf(req_id, sizeof(req_id), "chatcmpl-qw3-%llu", (unsigned long long)rseq);

    int cached_tokens = 0;
    double t0 = srv_now_ms();
    int sync_rc = srv_sync_prompt_plan(srv, &plan, fd, &cached_tokens, err, sizeof(err));
    double prefill_ms = srv_now_ms() - t0;
    int prefill_n = prompt->len - cached_tokens;

    if (sync_rc == -2) {
        pthread_mutex_unlock(&srv->infer_lock);
        srv_prompt_plan_free(&plan);
        req_params_free(&p);
        return;
    }
    if (sync_rc != 0) {
        pthread_mutex_unlock(&srv->infer_lock);
        send_http_error(fd, 500, "Internal Server Error", "server_error",
                        err[0] ? err : "prefill failed");
        srv_prompt_plan_free(&plan);
        req_params_free(&p);
        return;
    }

    if (p.stream) {
        if (!send_sse_headers(fd)) {
            pthread_mutex_unlock(&srv->infer_lock);
            srv_prompt_plan_free(&plan);
            req_params_free(&p);
            return;
        }
        if (!sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                 "assistant", NULL, 0, NULL, 0, NULL, 0, NULL)) {
            pthread_mutex_unlock(&srv->infer_lock);
            srv_prompt_plan_free(&plan);
            req_params_free(&p);
            return;
        }
    }

    /* Decode loop: filter out negative vision sentinels from repeat_penalty window */
    qw3_tokens recent = {0};
    qw3_tokens gen_raw = {0};
    for (int i = 0; i < prompt->len; i++) {
        if (prompt->v[i] >= 0) qw3_tokens_push(&recent, prompt->v[i]);
    }
    uint64_t rng = p.seed;
    int eos_id = qw3_token_eos(srv->engine);
    int max_gen = p.max_tokens;
    int avail_ctx = srv->cfg.ctx_size - prompt->len - 1;
    if (avail_ctx < 0) avail_ctx = 0;
    if (max_gen > avail_ctx) max_gen = avail_ctx;

    /* State for Inkling vs Qwen3 output parsing */
    enum { INK_HIDDEN, INK_SPEAKER, INK_TEXT, INK_THINK, INK_CALL } ink_state = INK_HIDDEN;
    strbuf ink_speaker, ink_args;
    sb_init(&ink_speaker);
    sb_init(&ink_args);

    bool qwen_in_think = (!srv->is_inkling && qw3_think_mode_enabled(p.think_mode));
    bool qwen_hold_tools = (!srv->is_inkling && p.has_tools);
    strbuf qwen_tag_buf;
    sb_init(&qwen_tag_buf);

    strbuf full_content, full_reasoning;
    strbuf pending_content, pending_reasoning;
    sb_init(&full_content);
    sb_init(&full_reasoning);
    sb_init(&pending_content);
    sb_init(&pending_reasoning);

    srv_tool_call_list tool_calls;
    memset(&tool_calls, 0, sizeof(tool_calls));

    int gen_tokens = 0;
    bool ended_at_eos = false;
    const char *finish_reason = "length";
    bool stream_ok = true;
    double tg0 = srv_now_ms();

    for (int step = 0; step < max_gen; step++) {
        if (client_disconnected(fd)) {
            stream_ok = false;
            break;
        }
        int tok = sample_next_token(srv->session, &p, &rng, &recent);
        if (tok < 0) {
            finish_reason = "stop";
            break;
        }
        gen_tokens++;
        qw3_tokens_push(&recent, tok);
        qw3_tokens_push(&gen_raw, tok);

        if (tok == eos_id) {
            /* Advance session past EOS so exact next-turn continuations can reuse KV */
            if (qw3_session_eval(srv->session, tok, err, sizeof(err)) == 0) {
                ended_at_eos = true;
            }
            finish_reason = tool_calls.count > 0 ? "tool_calls" : "stop";
            break;
        }

        size_t tlen = 0;
        char *tstr = srv_token_decoded_text(srv->engine, tok, &tlen);

        if (srv->is_inkling) {
            if (tok == srv->ink_model) {
                ink_speaker.len = 0;
                if (ink_speaker.p) ink_speaker.p[0] = '\0';
                ink_args.len = 0;
                if (ink_args.p) ink_args.p[0] = '\0';
                ink_state = INK_SPEAKER;
            } else if (tok == srv->ink_end) {
                if (ink_state == INK_CALL && tool_calls.count < QW3_SRV_MAX_TOOL_CALLS) {
                    srv_tool_call *tc = &tool_calls.calls[tool_calls.count];
                    snprintf(tc->id, sizeof(tc->id), "call_%d", tool_calls.count);
                    const char *fn_name = (ink_speaker.p && ink_speaker.len > 0) ? ink_speaker.p : "tool";
                    char *unwrapped_args = NULL;
                    if (ink_args.p && ink_args.len > 0) {
                        jval *env = json_parse(ink_args.p, ink_args.len);
                        if (env && env->type == J_OBJ) {
                            const char *en = jobj_str(env, "name");
                            if (en && en[0] && (!ink_speaker.p || ink_speaker.len == 0)) fn_name = en;
                            const jval *ea = jobj_get(env, "args");
                            if (!ea) ea = jobj_get(env, "arguments");
                            if (ea && ea->raw && ea->raw_len > 0) {
                                unwrapped_args = (char *)malloc(ea->raw_len + 1);
                                if (unwrapped_args) {
                                    memcpy(unwrapped_args, ea->raw, ea->raw_len);
                                    unwrapped_args[ea->raw_len] = '\0';
                                }
                            }
                        }
                        jval_free(env);
                    }
                    snprintf(tc->name, sizeof(tc->name), "%s", fn_name);
                    tc->arguments = unwrapped_args
                        ? unwrapped_args
                        : strdup((ink_args.p && ink_args.len > 0) ? ink_args.p : "{}");
                    if (p.stream) {
                        stream_ok = sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                                        NULL, NULL, 0, NULL, 0,
                                                        tc, tool_calls.count, NULL);
                    }
                    tool_calls.count++;
                }
                ink_state = INK_HIDDEN;
            } else if (ink_state == INK_SPEAKER && tok == srv->ink_think) {
                ink_state = INK_THINK;
            } else if (ink_state == INK_SPEAKER && (tok == srv->ink_text || tok == srv->ink_code)) {
                ink_state = INK_TEXT;
            } else if (ink_state == INK_SPEAKER && tok == srv->ink_call) {
                ink_state = INK_CALL;
            } else if (tstr && tlen >= 4 && !memcmp(tstr, "<|", 2) && !memcmp(tstr + tlen - 2, "|>", 2)) {
                ink_state = INK_HIDDEN;
            } else if (tstr && tlen > 0) {
                if (ink_state == INK_SPEAKER) {
                    sb_append_n(&ink_speaker, tstr, tlen);
                } else if (ink_state == INK_CALL) {
                    sb_append_n(&ink_args, tstr, tlen);
                } else if (ink_state == INK_THINK) {
                    sb_append_n(&pending_reasoning, tstr, tlen);
                } else if (ink_state == INK_TEXT) {
                    sb_append_n(&pending_content, tstr, tlen);
                }
            }
        } else if (tstr && tlen > 0) {
            /* Qwen3: split <think>...</think> into reasoning_content vs content */
            if (qwen_in_think) {
                sb_append_n(&qwen_tag_buf, tstr, tlen);
                const char *end_tag = strstr(qwen_tag_buf.p, "</think>");
                if (end_tag) {
                    size_t before = (size_t)(end_tag - qwen_tag_buf.p);
                    if (before > 0) sb_append_n(&pending_reasoning, qwen_tag_buf.p, before);
                    const char *after = end_tag + 8;
                    while (*after == '\n' || *after == '\r') after++;
                    if (*after) sb_append(&pending_content, after);
                    qwen_tag_buf.len = 0;
                    if (qwen_tag_buf.p) qwen_tag_buf.p[0] = '\0';
                    qwen_in_think = false;
                } else if (qwen_tag_buf.len > 8) {
                    size_t flush_n = utf8_complete_prefix(qwen_tag_buf.p, qwen_tag_buf.len - 8);
                    if (flush_n > 0) {
                        sb_append_n(&pending_reasoning, qwen_tag_buf.p, flush_n);
                        memmove(qwen_tag_buf.p, qwen_tag_buf.p + flush_n, qwen_tag_buf.len - flush_n + 1);
                        qwen_tag_buf.len -= flush_n;
                    }
                }
            } else {
                sb_append_n(&pending_content, tstr, tlen);
            }
        }
        free(tstr);

        /* Flush safe UTF-8 reasoning_content */
        if (pending_reasoning.len > 0) {
            size_t safe = utf8_complete_prefix(pending_reasoning.p, pending_reasoning.len);
            if (safe > 0) {
                sb_append_n(&full_reasoning, pending_reasoning.p, safe);
                if (p.stream && stream_ok) {
                    stream_ok = sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                                    NULL, NULL, 0,
                                                    pending_reasoning.p, safe,
                                                    NULL, 0, NULL);
                }
                memmove(pending_reasoning.p, pending_reasoning.p + safe,
                        pending_reasoning.len - safe + 1);
                pending_reasoning.len -= safe;
            }
        }

        /* Check stop sequences and flush content */
        bool hit_stop = false;
        if (pending_content.len > 0 && p.n_stops > 0) {
            size_t stop_at = find_stop_seq(&pending_content, &p);
            if (stop_at != SIZE_MAX) {
                hit_stop = true;
                pending_content.len = stop_at;
                pending_content.p[stop_at] = '\0';
            }
        }

        if (pending_content.len > 0 && !qwen_hold_tools) {
            size_t hold = (!hit_stop && p.max_stop_len > 1) ? (p.max_stop_len - 1) : 0;
            if (pending_content.len > hold) {
                size_t avail = pending_content.len - hold;
                size_t safe = hit_stop ? avail : utf8_complete_prefix(pending_content.p, avail);
                if (safe > 0) {
                    sb_append_n(&full_content, pending_content.p, safe);
                    if (p.stream && stream_ok) {
                        stream_ok = sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                                        NULL, pending_content.p, safe,
                                                        NULL, 0, NULL, 0, NULL);
                    }
                    memmove(pending_content.p, pending_content.p + safe,
                            pending_content.len - safe + 1);
                    pending_content.len -= safe;
                }
            }
        }

        if (!stream_ok) break;
        if (hit_stop) {
            finish_reason = tool_calls.count > 0 ? "tool_calls" : "stop";
            break;
        }

        if (step + 1 < max_gen) {
            if (qw3_session_eval(srv->session, tok, err, sizeof(err)) != 0) {
                break;
            }
        }
    }

    double gen_ms = srv_now_ms() - tg0;

    /* Flush remaining buffered text before hashing/recording the turn */
    if (qwen_tag_buf.len > 0) {
        sb_append_n(&pending_reasoning, qwen_tag_buf.p, qwen_tag_buf.len);
    }
    if (pending_reasoning.len > 0) {
        sb_append_n(&full_reasoning, pending_reasoning.p, pending_reasoning.len);
        if (p.stream && stream_ok) {
            stream_ok = sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                            NULL, NULL, 0,
                                            pending_reasoning.p, pending_reasoning.len,
                                            NULL, 0, NULL);
        }
    }

    if (qwen_hold_tools && pending_content.len > 0) {
        strbuf cleaned;
        sb_init(&cleaned);
        parse_qwen_tool_calls(pending_content.p, &tool_calls, &cleaned);
        if (cleaned.len > 0) {
            sb_append_n(&full_content, cleaned.p, cleaned.len);
            if (p.stream && stream_ok) {
                stream_ok = sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                                NULL, cleaned.p, cleaned.len,
                                                NULL, 0, NULL, 0, NULL);
            }
        }
        if (p.stream && stream_ok) {
            for (int i = 0; i < tool_calls.count; i++) {
                stream_ok = sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                                NULL, NULL, 0, NULL, 0,
                                                &tool_calls.calls[i], i, NULL);
                if (!stream_ok) break;
            }
        }
        if (tool_calls.count > 0) finish_reason = "tool_calls";
        sb_free(&cleaned);
    } else if (pending_content.len > 0) {
        sb_append_n(&full_content, pending_content.p, pending_content.len);
        if (p.stream && stream_ok) {
            stream_ok = sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                            NULL, pending_content.p, pending_content.len,
                                            NULL, 0, NULL, 0, NULL);
        }
    }

    /* Record exact generated tokens for seamless multi-turn KV prefix reuse */
    if (srv->is_inkling && ended_at_eos && gen_tokens > 0 &&
        srv->n_cached_turns < QW3_SRV_MAX_CACHED_TURNS &&
        gen_raw.len == gen_tokens) {
        cached_asst_turn *ct = &srv->cached_turns[srv->n_cached_turns++];
        memset(ct, 0, sizeof(*ct));
        ct->prompt_end = prompt->len;
        ct->turn_hash = hash_asst_turn_gen(&full_content, &tool_calls);
        for (int k = 0; k < gen_raw.len; k++) {
            qw3_tokens_push(&ct->raw_tokens, gen_raw.v[k]);
        }
    }
    pthread_mutex_unlock(&srv->infer_lock);

    if (plan.has_images) {
        fprintf(stderr,
            "qw3-server: %s prompt=%d (cached=%d, prefill=%d @ %.1f tok/s, imgs=%d vis=%.1fms) gen=%d @ %.1f tok/s finish=%s\n",
            req_id, prompt->len, cached_tokens, prefill_n,
            prefill_ms > 0.0 ? (1000.0 * prefill_n / prefill_ms) : 0.0,
            plan.n_images, plan.vis_encode_ms,
            gen_tokens, gen_ms > 0.0 ? (1000.0 * gen_tokens / gen_ms) : 0.0,
            finish_reason);
    } else {
        fprintf(stderr,
            "qw3-server: %s prompt=%d (cached=%d, prefill=%d @ %.1f tok/s) gen=%d @ %.1f tok/s finish=%s\n",
            req_id, prompt->len, cached_tokens, prefill_n,
            prefill_ms > 0.0 ? (1000.0 * prefill_n / prefill_ms) : 0.0,
            gen_tokens, gen_ms > 0.0 ? (1000.0 * gen_tokens / gen_ms) : 0.0,
            finish_reason);
    }

    if (p.stream) {
        if (stream_ok) {
            (void)sse_send_chat_delta(fd, req_id, created, srv->cfg.model_alias,
                                      NULL, NULL, 0, NULL, 0, NULL, 0, finish_reason);
            if (p.include_usage) {
                (void)sse_send_usage_chunk(fd, req_id, created, srv->cfg.model_alias,
                                           prompt->len, cached_tokens, gen_tokens,
                                           prefill_ms, prefill_n, gen_ms);
            }
            (void)send_all(fd, "data: [DONE]\n\n", 14);
        }
    } else {
        strbuf resp;
        sb_init(&resp);
        sb_append(&resp, "{\"id\":");
        sb_append_json_str(&resp, req_id);
        sb_appendf(&resp, ",\"object\":\"chat.completion\",\"created\":%lld,\"model\":",
                   (long long)created);
        sb_append_json_str(&resp, srv->cfg.model_alias);
        sb_append(&resp, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (full_content.len > 0 || tool_calls.count == 0) {
            sb_append_json_str_n(&resp, full_content.p ? full_content.p : "", full_content.len);
        } else {
            sb_append(&resp, "null");
        }
        if (full_reasoning.len > 0) {
            sb_append(&resp, ",\"reasoning_content\":");
            sb_append_json_str_n(&resp, full_reasoning.p, full_reasoning.len);
        }
        if (tool_calls.count > 0) {
            sb_append(&resp, ",\"tool_calls\":[");
            for (int i = 0; i < tool_calls.count; i++) {
                if (i > 0) sb_append_c(&resp, ',');
                sb_append(&resp, "{\"id\":");
                sb_append_json_str(&resp, tool_calls.calls[i].id);
                sb_append(&resp, ",\"type\":\"function\",\"function\":{\"name\":");
                sb_append_json_str(&resp, tool_calls.calls[i].name);
                sb_append(&resp, ",\"arguments\":");
                sb_append_json_str(&resp, tool_calls.calls[i].arguments ? tool_calls.calls[i].arguments : "{}");
                sb_append(&resp, "}}");
            }
            sb_append_c(&resp, ']');
        }
        sb_append(&resp, "},\"finish_reason\":");
        sb_append_json_str(&resp, finish_reason);
        sb_appendf(&resp,
            "}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
            "\"prompt_tokens_details\":{\"cached_tokens\":%d}},"
            "\"timings\":{\"cache_n\":%d,\"prompt_n\":%d,\"prompt_ms\":%.2f,\"prompt_per_second\":%.2f,"
            "\"predicted_n\":%d,\"predicted_ms\":%.2f,\"predicted_per_second\":%.2f}}\n",
            prompt->len, gen_tokens, prompt->len + gen_tokens, cached_tokens,
            cached_tokens, prefill_n, prefill_ms,
            prefill_ms > 0.0 ? (1000.0 * prefill_n / prefill_ms) : 0.0,
            gen_tokens, gen_ms, gen_ms > 0.0 ? (1000.0 * gen_tokens / gen_ms) : 0.0);
        send_http_json(fd, 200, "OK", resp.p, resp.len);
        sb_free(&resp);
    }

    srv_tool_calls_free(&tool_calls);
    sb_free(&ink_speaker);
    sb_free(&ink_args);
    sb_free(&qwen_tag_buf);
    sb_free(&full_content);
    sb_free(&full_reasoning);
    sb_free(&pending_content);
    sb_free(&pending_reasoning);
    qw3_tokens_free(&recent);
    qw3_tokens_free(&gen_raw);
    srv_prompt_plan_free(&plan);
    req_params_free(&p);
}

/* =========================================================================
 * /v1/completions handler
 * ========================================================================= */

static void handle_completions(srv_state *srv, int fd, const jval *body) {
    const jval *pv = jobj_get(body, "prompt");
    if (!pv || (pv->type != J_STR && pv->type != J_ARR)) {
        send_http_error(fd, 400, "Bad Request", "invalid_request_error",
                        "prompt must be a string or token array");
        return;
    }
    req_params p;
    parse_req_params(&srv->cfg, body, &p);

    qw3_tokens prompt = {0};
    if (pv->type == J_STR) {
        qw3_tokenize_text(srv->engine, pv->str, &prompt);
    } else {
        for (size_t i = 0; i < pv->len; i++) {
            if (pv->items[i] && pv->items[i]->type == J_NUM) {
                qw3_tokens_push(&prompt, (int)pv->items[i]->num);
            }
        }
    }
    if (prompt.len == 0 || prompt.len >= srv->cfg.ctx_size) {
        send_http_error(fd, 400, "Bad Request", "invalid_request_error",
                        "prompt is empty or exceeds context size");
        qw3_tokens_free(&prompt);
        req_params_free(&p);
        return;
    }

    char err[256] = {0};
    pthread_mutex_lock(&srv->infer_lock);
    uint64_t rseq = ++srv->req_counter;
    time_t created = time(NULL);
    char req_id[64];
    snprintf(req_id, sizeof(req_id), "cmpl-qw3-%llu", (unsigned long long)rseq);

    int session_pos = qw3_session_pos(srv->session);
    int common = qw3_session_common_prefix(srv->session, &prompt);
    int cached_tokens = (common == session_pos && prompt.len >= common) ? common : 0;
    int prefill_n = prompt.len - cached_tokens;

    double t0 = srv_now_ms();
    int sync_rc = qw3_session_sync_interruptible(srv->session, &prompt,
                                                 http_interrupt_cb, &fd,
                                                 err, sizeof(err));
    double prefill_ms = srv_now_ms() - t0;

    if (sync_rc != 0) {
        pthread_mutex_unlock(&srv->infer_lock);
        if (sync_rc != -2)
            send_http_error(fd, 500, "Internal Server Error", "server_error",
                            err[0] ? err : "prefill failed");
        qw3_tokens_free(&prompt);
        req_params_free(&p);
        return;
    }

    if (p.stream && !send_sse_headers(fd)) {
        pthread_mutex_unlock(&srv->infer_lock);
        qw3_tokens_free(&prompt);
        req_params_free(&p);
        return;
    }

    qw3_tokens recent = {0};
    qw3_tokens_copy(&recent, &prompt);
    uint64_t rng = p.seed;
    int eos_id = qw3_token_eos(srv->engine);
    int max_gen = p.max_tokens;
    int avail_ctx = srv->cfg.ctx_size - prompt.len - 1;
    if (avail_ctx < 0) avail_ctx = 0;
    if (max_gen > avail_ctx) max_gen = avail_ctx;

    strbuf full_text, pending;
    sb_init(&full_text);
    sb_init(&pending);

    int gen_tokens = 0;
    const char *finish_reason = "length";
    bool stream_ok = true;
    double tg0 = srv_now_ms();

    for (int step = 0; step < max_gen; step++) {
        if (client_disconnected(fd)) { stream_ok = false; break; }
        int tok = sample_next_token(srv->session, &p, &rng, &recent);
        if (tok < 0 || tok == eos_id) {
            if (tok == eos_id) {
                gen_tokens++;
                (void)qw3_session_eval(srv->session, tok, err, sizeof(err));
            }
            finish_reason = "stop";
            break;
        }
        gen_tokens++;
        qw3_tokens_push(&recent, tok);

        size_t tlen = 0;
        char *tstr = srv_token_decoded_text(srv->engine, tok, &tlen);
        if (tstr && tlen > 0) sb_append_n(&pending, tstr, tlen);
        free(tstr);

        bool hit_stop = false;
        if (pending.len > 0 && p.n_stops > 0) {
            size_t stop_at = find_stop_seq(&pending, &p);
            if (stop_at != SIZE_MAX) {
                hit_stop = true;
                pending.len = stop_at;
                pending.p[stop_at] = '\0';
            }
        }
        size_t hold = (!hit_stop && p.max_stop_len > 1) ? (p.max_stop_len - 1) : 0;
        if (pending.len > hold) {
            size_t avail = pending.len - hold;
            size_t safe = hit_stop ? avail : utf8_complete_prefix(pending.p, avail);
            if (safe > 0) {
                sb_append_n(&full_text, pending.p, safe);
                if (p.stream && stream_ok) {
                    strbuf chunk;
                    sb_init(&chunk);
                    sb_append(&chunk, "{\"id\":");
                    sb_append_json_str(&chunk, req_id);
                    sb_appendf(&chunk, ",\"object\":\"text_completion\",\"created\":%lld,\"model\":",
                               (long long)created);
                    sb_append_json_str(&chunk, srv->cfg.model_alias);
                    sb_append(&chunk, ",\"choices\":[{\"index\":0,\"text\":");
                    sb_append_json_str_n(&chunk, pending.p, safe);
                    sb_append(&chunk, ",\"finish_reason\":null}]}");
                    stream_ok = send_sse_data(fd, chunk.p, chunk.len);
                    sb_free(&chunk);
                }
                memmove(pending.p, pending.p + safe, pending.len - safe + 1);
                pending.len -= safe;
            }
        }
        if (!stream_ok) break;
        if (hit_stop) { finish_reason = "stop"; break; }
        if (step + 1 < max_gen && qw3_session_eval(srv->session, tok, err, sizeof(err)) != 0) break;
    }

    double gen_ms = srv_now_ms() - tg0;
    pthread_mutex_unlock(&srv->infer_lock);

    if (pending.len > 0) {
        sb_append_n(&full_text, pending.p, pending.len);
        if (p.stream && stream_ok) {
            strbuf chunk;
            sb_init(&chunk);
            sb_append(&chunk, "{\"id\":");
            sb_append_json_str(&chunk, req_id);
            sb_appendf(&chunk, ",\"object\":\"text_completion\",\"created\":%lld,\"model\":",
                       (long long)created);
            sb_append_json_str(&chunk, srv->cfg.model_alias);
            sb_append(&chunk, ",\"choices\":[{\"index\":0,\"text\":");
            sb_append_json_str_n(&chunk, pending.p, pending.len);
            sb_append(&chunk, ",\"finish_reason\":null}]}");
            stream_ok = send_sse_data(fd, chunk.p, chunk.len);
            sb_free(&chunk);
        }
    }

    if (p.stream) {
        if (stream_ok) {
            strbuf fin;
            sb_init(&fin);
            sb_append(&fin, "{\"id\":");
            sb_append_json_str(&fin, req_id);
            sb_appendf(&fin, ",\"object\":\"text_completion\",\"created\":%lld,\"model\":",
                       (long long)created);
            sb_append_json_str(&fin, srv->cfg.model_alias);
            sb_append(&fin, ",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":");
            sb_append_json_str(&fin, finish_reason);
            sb_append(&fin, "}]}");
            (void)send_sse_data(fd, fin.p, fin.len);
            sb_free(&fin);
            (void)send_all(fd, "data: [DONE]\n\n", 14);
        }
    } else {
        strbuf resp;
        sb_init(&resp);
        sb_append(&resp, "{\"id\":");
        sb_append_json_str(&resp, req_id);
        sb_appendf(&resp, ",\"object\":\"text_completion\",\"created\":%lld,\"model\":",
                   (long long)created);
        sb_append_json_str(&resp, srv->cfg.model_alias);
        sb_append(&resp, ",\"choices\":[{\"index\":0,\"text\":");
        sb_append_json_str_n(&resp, full_text.p ? full_text.p : "", full_text.len);
        sb_append(&resp, ",\"finish_reason\":");
        sb_append_json_str(&resp, finish_reason);
        sb_appendf(&resp,
            "}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
            "\"prompt_tokens_details\":{\"cached_tokens\":%d}},"
            "\"timings\":{\"cache_n\":%d,\"prompt_n\":%d,\"prompt_ms\":%.2f,\"prompt_per_second\":%.2f,"
            "\"predicted_n\":%d,\"predicted_ms\":%.2f,\"predicted_per_second\":%.2f}}\n",
            prompt.len, gen_tokens, prompt.len + gen_tokens, cached_tokens,
            cached_tokens, prefill_n, prefill_ms,
            prefill_ms > 0.0 ? (1000.0 * prefill_n / prefill_ms) : 0.0,
            gen_tokens, gen_ms, gen_ms > 0.0 ? (1000.0 * gen_tokens / gen_ms) : 0.0);
        send_http_json(fd, 200, "OK", resp.p, resp.len);
        sb_free(&resp);
    }

    sb_free(&full_text);
    sb_free(&pending);
    qw3_tokens_free(&recent);
    qw3_tokens_free(&prompt);
    req_params_free(&p);
}

/* =========================================================================
 * /tokenize handler
 * ========================================================================= */

static void handle_tokenize(srv_state *srv, int fd, const jval *body) {
    const char *text = jobj_str(body, "content");
    if (!text) text = jobj_str(body, "text");
    if (!text) {
        send_http_error(fd, 400, "Bad Request", "invalid_request_error",
                        "expected string field 'content' or 'text'");
        return;
    }
    qw3_tokens toks = {0};
    pthread_mutex_lock(&srv->infer_lock);
    qw3_tokenize_text(srv->engine, text, &toks);
    pthread_mutex_unlock(&srv->infer_lock);

    strbuf resp;
    sb_init(&resp);
    sb_append(&resp, "{\"tokens\":[");
    for (int i = 0; i < toks.len; i++) {
        if (i > 0) sb_append_c(&resp, ',');
        sb_appendf(&resp, "%d", toks.v[i]);
    }
    sb_appendf(&resp, "],\"count\":%d}\n", toks.len);
    send_http_json(fd, 200, "OK", resp.p, resp.len);
    sb_free(&resp);
    qw3_tokens_free(&toks);
}

/* =========================================================================
 * Per-connection worker thread
 * ========================================================================= */

typedef struct {
    srv_state *srv;
    int fd;
} conn_arg;

static bool check_auth(const srv_state *srv, const http_request *req) {
    if (!srv->cfg.api_key || !srv->cfg.api_key[0]) return true;
    if (strncmp(req->auth, "Bearer ", 7) != 0) return false;
    return strcmp(req->auth + 7, srv->cfg.api_key) == 0;
}

static void *handle_connection(void *arg) {
    conn_arg *ca = (conn_arg *)arg;
    srv_state *srv = ca->srv;
    int fd = ca->fd;
    free(ca);

    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    http_request req;
    int rc = read_http_request(fd, &req);
    if (rc != 0) {
        if (rc == -3)
            send_http_error(fd, 411, "Length Required", "invalid_request_error",
                            "chunked request bodies are not supported");
        else if (rc == -4)
            send_http_error(fd, 413, "Payload Too Large", "invalid_request_error",
                            "request body exceeds 32 MiB limit");
        else if (rc == -2)
            send_http_error(fd, 400, "Bad Request", "invalid_request_error", "malformed HTTP request");
        close(fd);
        return NULL;
    }

    if (strcmp(req.method, "OPTIONS") == 0) {
        static const char opt_resp[] =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n\r\n";
        (void)send_all(fd, opt_resp, sizeof(opt_resp) - 1);
        http_request_free(&req);
        close(fd);
        return NULL;
    }

    if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        strbuf resp;
        sb_init(&resp);
        sb_append(&resp, "{\"status\":\"ok\",\"model\":");
        sb_append_json_str(&resp, srv->cfg.model_alias);
        sb_appendf(&resp, ",\"ctx_size\":%d,\"backend\":", srv->cfg.ctx_size);
        sb_append_json_str(&resp, qw3_backend_name(srv->cfg.backend));
        sb_appendf(&resp, ",\"vision\":%s", srv->vision ? "true" : "false");
#ifndef QW3_NO_METAL
        if (srv->vision) {
            sb_append(&resp, ",\"vision_arch\":");
            sb_append_json_str(&resp, qw3_vision_arch(srv->vision));
        }
#endif
        sb_append(&resp, "}\n");
        send_http_json(fd, 200, "OK", resp.p, resp.len);
        sb_free(&resp);
        http_request_free(&req);
        close(fd);
        return NULL;
    }

    if (!check_auth(srv, &req)) {
        send_http_error(fd, 401, "Unauthorized", "authentication_error", "invalid or missing API key");
        http_request_free(&req);
        close(fd);
        return NULL;
    }

    if (strcmp(req.method, "GET") == 0 &&
        (strcmp(req.path, "/v1/models") == 0 || strcmp(req.path, "/models") == 0)) {
        strbuf resp;
        sb_init(&resp);
        sb_append(&resp, "{\"object\":\"list\",\"data\":[{\"id\":");
        sb_append_json_str(&resp, srv->cfg.model_alias);
        sb_appendf(&resp, ",\"object\":\"model\",\"created\":%lld,\"owned_by\":\"qw3\"}]}\n",
                   (long long)srv->started_at);
        send_http_json(fd, 200, "OK", resp.p, resp.len);
        sb_free(&resp);
    } else if (strcmp(req.method, "POST") == 0 &&
               (strcmp(req.path, "/v1/chat/completions") == 0 ||
                strcmp(req.path, "/chat/completions") == 0 ||
                strcmp(req.path, "/v1/completions") == 0 ||
                strcmp(req.path, "/completions") == 0 ||
                strcmp(req.path, "/tokenize") == 0)) {
        jval *body = json_parse(req.body ? req.body : "", req.body_len);
        if (!body || body->type != J_OBJ) {
            send_http_error(fd, 400, "Bad Request", "invalid_request_error", "invalid JSON object body");
        } else if (strcmp(req.path, "/v1/chat/completions") == 0 ||
                   strcmp(req.path, "/chat/completions") == 0) {
            handle_chat_completions(srv, fd, body);
        } else if (strcmp(req.path, "/v1/completions") == 0 ||
                   strcmp(req.path, "/completions") == 0) {
            handle_completions(srv, fd, body);
        } else {
            handle_tokenize(srv, fd, body);
        }
        jval_free(body);
    } else {
        send_http_error(fd, 404, "Not Found", "invalid_request_error", "unknown endpoint");
    }

    http_request_free(&req);
    close(fd);
    return NULL;
}

/* =========================================================================
 * CLI argument parsing and server startup
 * ========================================================================= */

static void print_usage(void) {
    printf(
        "qw3-server - OpenAI-compatible HTTP API server for qw3 (Inkling 2 Mini & Qwen3)\n\n"
        "Usage:\n"
        "  qw3-server -m MODEL [options]\n\n"
        "Options:\n"
        "  -m, --model PATH       GGUF model path (default: ./qw3.gguf)\n"
        "  --mmproj PATH          Vision projector GGUF (Inkling-2-Mini inkling2v)\n"
        "  --vision-max-tokens N  Max visual tokens per image (default: 4096)\n"
        "  --alias NAME           Model ID returned by /v1/models (default: filename)\n"
        "  --host ADDR            Bind address (default: 127.0.0.1)\n"
        "  --port N               Listen port (default: 8080)\n"
        "  -c, --ctx, --ctx-size N Context size in tokens (default: 32768)\n"
        "  --metal | --cpu        Select backend (default: Metal on macOS)\n"
        "  --ngl N                Metal layers to offload (Qwen3 models)\n"
        "  --kv-f16 | --kv-f32 | --kv-q8\n"
        "                         Metal KV cache format (Inkling defaults to FP16)\n"
        "  -ctk TYPE -ctv TYPE    Explicit Metal K/V cache types (f16, f32, q8_0)\n"
        "  -t, --threads N        CPU helper threads\n"
        "  --warm-weights         Touch mapped weight pages on startup\n"
        "  --nothink              Default thinking mode off (overridable per request)\n"
        "  --think-effort F       Default Inkling thinking effort level 0.0..0.99\n"
        "  --temp N               Default temperature (default: 0.6)\n"
        "  --top-p N              Default top-p (default: 0.95)\n"
        "  --top-k N              Default top-k (default: 40)\n"
        "  --min-p N              Default min-p (default: 0.0)\n"
        "  --repeat-penalty N     Default repetition penalty (default: 1.06)\n"
        "  --repeat-last-n N      Default repetition window (default: 1024)\n"
        "  --max-tokens N         Default max completion tokens (default: 2048)\n"
        "  --api-key KEY          Require Authorization: Bearer KEY\n"
        "  -h, --help             Show this help\n");
}

static void derive_alias(const char *path, char *out, size_t out_sz) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    size_t len = strlen(base);
    if (len > 5 && strcasecmp(base + len - 5, ".gguf") == 0) len -= 5;
    if (len >= out_sz) len = out_sz - 1;
    memcpy(out, base, len);
    out[len] = '\0';
}

static int bind_listen_socket(const char *host, int port) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    char pbuf[16];
    snprintf(pbuf, sizeof(pbuf), "%d", port);
    if (getaddrinfo(host, pbuf, &hints, &res) != 0) return -1;

    int lfd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 32) == 0) {
            lfd = fd;
            break;
        }
        close(fd);
    }
    freeaddrinfo(res);
    return lfd;
}

int main(int argc, char **argv) {
    srv_state srv;
    memset(&srv, 0, sizeof(srv));
    srv.cfg.model_path = "./qw3.gguf";
    srv.cfg.host = "127.0.0.1";
    srv.cfg.port = 8080;
    srv.cfg.ctx_size = 32768;
#ifdef QW3_NO_METAL
    srv.cfg.backend = QW3_BACKEND_CPU;
#else
    srv.cfg.backend = QW3_BACKEND_METAL;
#endif
    srv.cfg.ngl = -1;
    srv.cfg.default_think = QW3_THINK_ON;
    srv.cfg.default_temp = 0.6f;
    srv.cfg.default_top_p = 0.95f;
    srv.cfg.default_top_k = 40;
    srv.cfg.default_min_p = 0.0f;
    srv.cfg.default_repeat_penalty = 1.06f;
    srv.cfg.default_repeat_last_n = 1024;
    srv.cfg.default_max_tokens = 2048;

    const char *ctk = NULL, *ctv = NULL, *kv_alias = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage();
            return 0;
        } else if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            srv.cfg.model_path = argv[++i];
        } else if (!strcmp(argv[i], "--mmproj") && i + 1 < argc) {
            srv.cfg.mmproj_path = argv[++i];
        } else if (!strcmp(argv[i], "--vision-max-tokens") && i + 1 < argc) {
            srv.cfg.vision_max_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--alias") && i + 1 < argc) {
            snprintf(srv.cfg.model_alias, sizeof(srv.cfg.model_alias), "%s", argv[++i]);
        } else if (!strcmp(argv[i], "--host") && i + 1 < argc) {
            srv.cfg.host = argv[++i];
        } else if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            srv.cfg.port = atoi(argv[++i]);
        } else if ((!strcmp(argv[i], "-c") || !strcmp(argv[i], "--ctx") ||
                    !strcmp(argv[i], "--ctx-size")) && i + 1 < argc) {
            srv.cfg.ctx_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--metal")) {
            srv.cfg.backend = QW3_BACKEND_METAL;
        } else if (!strcmp(argv[i], "--cpu")) {
            srv.cfg.backend = QW3_BACKEND_CPU;
        } else if (!strcmp(argv[i], "--ngl") && i + 1 < argc) {
            srv.cfg.ngl = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--kv-f16")) {
            kv_alias = "f16";
        } else if (!strcmp(argv[i], "--kv-f32")) {
            kv_alias = "f32";
        } else if (!strcmp(argv[i], "--kv-q8")) {
            kv_alias = "q8_0";
        } else if (!strcmp(argv[i], "-ctk") && i + 1 < argc) {
            ctk = argv[++i];
        } else if (!strcmp(argv[i], "-ctv") && i + 1 < argc) {
            ctv = argv[++i];
        } else if ((!strcmp(argv[i], "-t") || !strcmp(argv[i], "--threads")) && i + 1 < argc) {
            srv.cfg.n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--warm-weights")) {
            srv.cfg.warm_weights = true;
        } else if (!strcmp(argv[i], "--nothink")) {
            srv.cfg.default_think = QW3_THINK_NONE;
            srv.cfg.default_think_effort_set = false;
        } else if (!strcmp(argv[i], "--think-effort") && i + 1 < argc) {
            const char *val = argv[++i];
            char *end = NULL;
            float effort = strtof(val, &end);
            if (!val[0] || *end || !isfinite(effort) || effort < 0.0f || effort > 0.99f) {
                fprintf(stderr, "qw3-server: --think-effort must be a float in [0.0, 0.99]\n");
                return 1;
            }
            srv.cfg.default_think_effort = effort;
            srv.cfg.default_think_effort_set = true;
            srv.cfg.default_think = (effort <= 0.0f) ? QW3_THINK_NONE : QW3_THINK_ON;
        } else if (!strcmp(argv[i], "--temp") && i + 1 < argc) {
            srv.cfg.default_temp = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) {
            srv.cfg.default_top_p = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) {
            srv.cfg.default_top_k = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) {
            srv.cfg.default_min_p = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) {
            srv.cfg.default_repeat_penalty = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--repeat-last-n") && i + 1 < argc) {
            srv.cfg.default_repeat_last_n = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) {
            srv.cfg.default_max_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--api-key") && i + 1 < argc) {
            srv.cfg.api_key = argv[++i];
        } else {
            fprintf(stderr, "qw3-server: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (srv.cfg.port <= 0 || srv.cfg.port > 65535 || srv.cfg.ctx_size < 128) {
        fprintf(stderr, "qw3-server: invalid port or ctx size\n");
        return 1;
    }
    if (!srv.cfg.model_alias[0]) {
        derive_alias(srv.cfg.model_path, srv.cfg.model_alias, sizeof(srv.cfg.model_alias));
    }

    if (srv.cfg.ngl >= 0) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", srv.cfg.ngl);
        setenv("QW3_METAL_NGL", buf, 1);
    }
    if (kv_alias) {
        ctk = kv_alias;
        ctv = kv_alias;
    }
    if (ctk || ctv) {
        if (!ctk || !ctv || strcmp(ctk, ctv) != 0) {
            fprintf(stderr, "qw3-server: -ctk and -ctv must match\n");
            return 1;
        }
        if (strcmp(ctk, "f16") == 0) {
            setenv("QW3_METAL_KV_F16", "1", 1);
            setenv("QW3_METAL_KV_Q8_0", "0", 1);
        } else if (strcmp(ctk, "f32") == 0) {
            setenv("QW3_METAL_KV_F16", "0", 1);
            setenv("QW3_METAL_KV_Q8_0", "0", 1);
        } else if (strcmp(ctk, "q8_0") == 0) {
            setenv("QW3_METAL_KV_F16", "0", 1);
            setenv("QW3_METAL_KV_Q8_0", "1", 1);
        } else {
            fprintf(stderr, "qw3-server: unsupported KV type '%s'\n", ctk);
            return 1;
        }
    }

    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = srv_handle_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    qw3_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = srv.cfg.model_path;
    opt.backend = srv.cfg.backend;
    opt.n_threads = srv.cfg.n_threads;
    opt.warm_weights = srv.cfg.warm_weights;
    opt.enable_inkling = true;

    if (qw3_engine_open(&srv.engine, &opt) != 0) {
        fprintf(stderr, "qw3-server: failed to load model %s\n", srv.cfg.model_path);
        return 1;
    }
    if (qw3_session_create(&srv.session, srv.engine, srv.cfg.ctx_size) != 0) {
        fprintf(stderr, "qw3-server: failed to create session (ctx=%d)\n", srv.cfg.ctx_size);
        qw3_engine_close(srv.engine);
        return 1;
    }

    srv.is_inkling = qw3_engine_is_inkling(srv.engine);
    if (srv.is_inkling) {
        srv.ink_user = qw3_inkling_chat_control_id(srv.engine, "<|message_user|>");
        srv.ink_tool = qw3_inkling_chat_control_id(srv.engine, "<|message_tool|>");
        srv.ink_model = qw3_inkling_chat_control_id(srv.engine, "<|message_model|>");
        srv.ink_text = qw3_inkling_chat_control_id(srv.engine, "<|content_text|>");
        srv.ink_image = qw3_inkling_chat_control_id(srv.engine, "<|content_image|>");
        srv.ink_code = qw3_inkling_chat_control_id(srv.engine, "<|content_code|>");
        srv.ink_think = qw3_inkling_chat_control_id(srv.engine, "<|content_thinking|>");
        srv.ink_call = qw3_inkling_chat_control_id(srv.engine, "<|content_invoke_tool_json|>");
        srv.ink_end = qw3_inkling_chat_control_id(srv.engine, "<|end_message|>");
        srv.ink_eos = qw3_token_eos(srv.engine);
    }

    if (srv.cfg.mmproj_path && srv.cfg.mmproj_path[0]) {
#ifdef QW3_NO_METAL
        fprintf(stderr, "qw3-server: --mmproj requires the Metal build (qw3-server)\n");
        qw3_session_free(srv.session);
        qw3_engine_close(srv.engine);
        return 1;
#else
        char verr[256] = {0};
        if (qw3_vision_open(&srv.vision, srv.cfg.mmproj_path,
                            srv.cfg.vision_max_tokens, verr, sizeof(verr)) != 0) {
            fprintf(stderr, "qw3-server: failed to load mmproj %s: %s\n",
                    srv.cfg.mmproj_path, verr);
            qw3_session_free(srv.session);
            qw3_engine_close(srv.engine);
            return 1;
        }
        int eng_dim = qw3_engine_embd_dim(srv.engine);
        int vis_dim = qw3_vision_embd_dim(srv.vision);
        if (eng_dim > 0 && vis_dim != eng_dim) {
            fprintf(stderr, "qw3-server: mmproj projection_dim (%d) != model embd_dim (%d)\n",
                    vis_dim, eng_dim);
            qw3_vision_close(srv.vision);
            qw3_session_free(srv.session);
            qw3_engine_close(srv.engine);
            return 1;
        }
#endif
    }

    qw3_context_memory mem = qw3_engine_context_memory_estimate(srv.engine, srv.cfg.ctx_size);
    pthread_mutex_init(&srv.infer_lock, NULL);
    srv.started_at = time(NULL);

    int lfd = bind_listen_socket(srv.cfg.host, srv.cfg.port);
    if (lfd < 0) {
        fprintf(stderr, "qw3-server: failed to bind %s:%d: %s\n",
                srv.cfg.host, srv.cfg.port, strerror(errno));
#ifndef QW3_NO_METAL
        qw3_vision_close(srv.vision);
#endif
        qw3_session_free(srv.session);
        qw3_engine_close(srv.engine);
        return 1;
    }

    const char *kv_q8_env = (srv.cfg.backend == QW3_BACKEND_METAL) ? getenv("QW3_METAL_KV_Q8_0") : NULL;
    const char *kv_f16_env = (srv.cfg.backend == QW3_BACKEND_METAL) ? getenv("QW3_METAL_KV_F16") : NULL;
    const char *kv_fmt = srv.cfg.backend != QW3_BACKEND_METAL ? "f32" :
        ((kv_q8_env && strcmp(kv_q8_env, "0") != 0) ? "q8_0" :
         (srv.is_inkling || (kv_f16_env && strcmp(kv_f16_env, "0") != 0)) ? "f16" : "f32");
    fprintf(stderr,
        "qw3-server: listening on http://%s:%d  model=%s  backend=%s  ctx=%d  kv=%s(%.1f MiB)",
        srv.cfg.host, srv.cfg.port, srv.cfg.model_alias,
        qw3_backend_name(srv.cfg.backend), srv.cfg.ctx_size, kv_fmt,
        (double)mem.total_bytes / (1024.0 * 1024.0));
#ifndef QW3_NO_METAL
    if (srv.vision) {
        fprintf(stderr, "  vision=%s(max_tok=%d)",
                qw3_vision_arch(srv.vision), qw3_vision_max_tokens(srv.vision));
    }
#endif
    fprintf(stderr, "\n");
    fprintf(stderr,
        "qw3-server: endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, "
        "POST /v1/completions, POST /tokenize\n");

    while (!g_stop_requested) {
        struct pollfd pfd;
        pfd.fd = lfd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int r = poll(&pfd, 1, 250);
        if (r <= 0) continue;
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;
        conn_arg *ca = (conn_arg *)malloc(sizeof(*ca));
        if (!ca) { close(cfd); continue; }
        ca->srv = &srv;
        ca->fd = cfd;
        pthread_t th;
        if (pthread_create(&th, NULL, handle_connection, ca) == 0) {
            pthread_detach(th);
        } else {
            free(ca);
            close(cfd);
        }
    }

    fprintf(stderr, "\nqw3-server: shutting down...\n");
    close(lfd);
    pthread_mutex_lock(&srv.infer_lock);
    srv_clear_cached_turns(&srv);
    srv_clear_vis_cache(&srv);
#ifndef QW3_NO_METAL
    qw3_vision_close(srv.vision);
#endif
    qw3_session_free(srv.session);
    qw3_engine_close(srv.engine);
    pthread_mutex_unlock(&srv.infer_lock);
    pthread_mutex_destroy(&srv.infer_lock);
    return 0;
}
