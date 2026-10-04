/* Optional process supervisor. Included after the ordinary agent/TUI code.
 * Model-owning children share neither pointers, tokens nor Metal resources.
 * The private IPC is versioned and separate from stdout/stderr. */
#define HYBRID_MAGIC 0x48594231u
#define HYBRID_MAX_FRAME (2u * 1024u * 1024u)
enum { H_RUN = 1, H_CONTINUE, H_COMMAND, H_CANCEL, H_STOP,
       H_READY = 100, H_OUTPUT, H_STATUS, H_ACTIVITY, H_PROGRESS, H_TIMING,
       H_DONE, H_DELEGATE, H_ERROR, H_BUDGET };
typedef struct { uint32_t magic, type, bytes; } hybrid_header;
typedef struct { int rc, tokens, pending, generated; } hybrid_done;
typedef struct { int current, total; double tps; char phase[32]; } hybrid_progress;
typedef struct {
    int cached, prompt, prefill, generated;
    double prefill_ms, pp, gen_ms, tg;
    char backend[24];
} hybrid_timing;
typedef struct { int mode, force; uint32_t job_bytes; } hybrid_run_options;
typedef struct {
    uint32_t magic, version;
    agent_turn_state turn;
    sample_opts sample;
    int pending, compact_retry, last_read_next, generated, transcript_tokens;
    uint32_t request_bytes, last_read_bytes;
    struct { int chunks, lines; uint32_t bytes; } source[QW3_AGENT_SOURCE_READ_TRACKED];
} hybrid_runtime;
typedef struct { pid_t pid; int fd, logs; bool dense; } hybrid_process;
typedef struct {
    char *executable, *primary_model, *dense_model, *root, *primary_store;
    char *active, *lock, *job;
    hybrid_process process;
    int phase;
    bool pending;
    double started;
    int generated;
    char activity[64];
} hybrid_supervisor;
enum { H_PHASE_PRIMARY = 1, H_PHASE_EXPERT, H_PHASE_RETURN, H_PHASE_DONE };

extern char **environ;

static bool hybrid_io(int fd, void *buffer, size_t bytes, bool writing) {
    char *p = buffer;
    while (bytes) {
        ssize_t n = writing ? write(fd, p, bytes) : read(fd, p, bytes);
        if (n > 0) { p += n; bytes -= (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd pollfd = {.fd = fd, .events = writing ? POLLOUT : POLLIN};
            if (poll(&pollfd, 1, 5000) > 0) continue;
        }
        return false;
    }
    return true;
}

static bool hybrid_send(int fd, uint32_t type, const void *data, size_t len) {
    if (fd < 0 || len > HYBRID_MAX_FRAME) return false;
    hybrid_header header = {HYBRID_MAGIC, type, (uint32_t)len};
    return hybrid_io(fd, &header, sizeof(header), true) &&
           (!len || hybrid_io(fd, (void *)data, len, true));
}

static char *hybrid_receive(int fd, uint32_t *type, size_t *len) {
    hybrid_header header;
    if (!hybrid_io(fd, &header, sizeof(header), false) ||
        header.magic != HYBRID_MAGIC || header.bytes > HYBRID_MAX_FRAME) return NULL;
    char *data = malloc((size_t)header.bytes + 1);
    if (!data) return NULL;
    if (header.bytes && !hybrid_io(fd, data, header.bytes, false)) { free(data); return NULL; }
    data[header.bytes] = '\0';
    *type = header.type;
    *len = header.bytes;
    return data;
}

static bool hybrid_atomic(const char *path, const void *data, size_t bytes) {
    char temporary[PATH_MAX];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.%ld", path, (long)getpid()) >= (int)sizeof(temporary)) return false;
    int fd = open(temporary, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    bool ok = hybrid_io(fd, (void *)data, bytes, true) && fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    if (ok) ok = rename(temporary, path) == 0;
    if (ok) {
        char directory[PATH_MAX]; snprintf(directory, sizeof(directory), "%s", path);
        char *slash = strrchr(directory, '/');
        if (slash) *slash = '\0'; else snprintf(directory, sizeof(directory), ".");
        int parent = open(directory, O_RDONLY);
        if (parent < 0 || fsync(parent) != 0) ok = false;
        if (parent >= 0) close(parent);
    }
    if (!ok) unlink(temporary);
    return ok;
}

static const char *hybrid_mode_name(agent_expert_mode mode) {
    return mode == AGENT_EXPERT_AUTO ? "auto" : mode == AGENT_EXPERT_MANUAL ? "manual" : "off";
}

static char *hybrid_absolute(const char *path) {
    if (!path) return NULL;
    char resolved[PATH_MAX];
    if (realpath(path, resolved)) return agent_strdup(resolved);
    if (path[0] == '/') return agent_strdup(path);
    char cwd[PATH_MAX];
    return getcwd(cwd, sizeof(cwd)) ? path_join(cwd, path) : NULL;
}

static char *hybrid_runtime_path(agent_state *a) {
    return store_path_ext(a, a->cfg.conversation, ".hybrid-state");
}

/* A compact atomic bundle commits transcript, message ledger and turn state
 * together. KV has its own atomic/checksummed file and falls back to this text.
 * A crash while rewriting ordinary store files cannot destroy the last bundle. */
typedef struct { uint32_t magic, version, bytes[5]; uint64_t hash; } hybrid_bundle;
static const char *hybrid_bundle_extensions[] = {".qw3a", ".msgs", ".txt", ".meta", ".hybrid-state"};
static bool hybrid_commit_bundle(agent_state *a) {
    if (!a->engine) return true; /* model-free state selftest */
    hybrid_bundle header = {.magic = HYBRID_MAGIC, .version = 1};
    strbuf body = {0}; bool ok = true;
    for (int i = 0; i < 5; i++) {
        char *path = store_path_ext(a, a->cfg.conversation, hybrid_bundle_extensions[i]);
        size_t len = 0; char *bytes = path ? read_file_text(path, &len) : NULL; free(path);
        if (!bytes || len > 32u*1024u*1024u || body.len + len > 32u*1024u*1024u) ok = false;
        if (ok) { header.bytes[i] = (uint32_t)len; sb_append_n(&body, bytes, len); }
        free(bytes); if (!ok) break;
    }
    header.hash = agent_hash_update(1469598103934665603ull, body.p, body.len);
    strbuf bundle = {0}; sb_append_n(&bundle, (char *)&header, sizeof(header));
    sb_append_n(&bundle, body.p, body.len);
    char *path = store_path_ext(a, a->cfg.conversation, ".hybrid-recovery");
    ok = ok && path && hybrid_atomic(path, bundle.p, bundle.len);
    free(path); sb_free(&body); sb_free(&bundle); return ok;
}

static void agent_hybrid_restore(agent_state *a) {
    char *path = store_path_ext(a, a->cfg.conversation, ".hybrid-recovery");
    size_t len = 0; char *bundle = path ? read_file_text(path, &len) : NULL; free(path);
    if (!bundle) return;
    hybrid_bundle header; bool valid = len >= sizeof(header) && len <= 32u*1024u*1024u+sizeof(header);
    size_t expected = sizeof(header);
    if (valid) {
        memcpy(&header, bundle, sizeof(header));
        for (int i = 0; i < 5; i++) expected += header.bytes[i];
        valid = header.magic == HYBRID_MAGIC && header.version == 1 && expected == len &&
            header.hash == agent_hash_update(1469598103934665603ull, bundle + sizeof(header), len - sizeof(header));
    }
    if (valid) {
        size_t pos = sizeof(header);
        for (int i = 0; i < 5; i++) {
            path = store_path_ext(a, a->cfg.conversation, hybrid_bundle_extensions[i]);
            if (!path || !hybrid_atomic(path, bundle + pos, header.bytes[i])) valid = false;
            free(path); pos += header.bytes[i];
            if (!valid) break;
        }
    }
    if (!valid) agent_statusf(a, "agent: hybrid recovery bundle unavailable; using ordinary transcript fallback\n");
    free(bundle);
}

static bool hybrid_save_runtime(agent_state *a) {
    hybrid_runtime state = {0};
    state.magic = HYBRID_MAGIC; state.version = 2;
    state.turn = a->turn; state.sample = a->cfg.sample;
    state.pending = a->edit_verification_pending;
    state.compact_retry = a->compact_retry_after_tokens;
    state.last_read_next = a->last_read_next;
    state.generated = a->hybrid_generated;
    state.transcript_tokens = a->transcript.len;
    state.request_bytes = (uint32_t)strlen(a->turn_request ? a->turn_request : "");
    state.last_read_bytes = (uint32_t)strlen(a->last_read_path ? a->last_read_path : "");
    for (int i = 0; i < QW3_AGENT_SOURCE_READ_TRACKED; i++) {
        state.source[i].chunks = a->source_reads[i].chunks;
        state.source[i].lines = a->source_reads[i].lines;
        state.source[i].bytes = (uint32_t)strlen(a->source_reads[i].path ? a->source_reads[i].path : "");
    }
    strbuf bytes = {0};
    sb_append_n(&bytes, (char *)&state, sizeof(state));
    sb_append_n(&bytes, a->turn_request ? a->turn_request : "", state.request_bytes);
    sb_append_n(&bytes, a->last_read_path ? a->last_read_path : "", state.last_read_bytes);
    for (int i = 0; i < QW3_AGENT_SOURCE_READ_TRACKED; i++)
        sb_append_n(&bytes, a->source_reads[i].path ? a->source_reads[i].path : "", state.source[i].bytes);
    char *path = hybrid_runtime_path(a);
    bool ok = path && hybrid_atomic(path, bytes.p, bytes.len) && hybrid_commit_bundle(a);
    free(path); sb_free(&bytes);
    return ok;
}

static bool hybrid_load_runtime(agent_state *a) {
    char *path = hybrid_runtime_path(a);
    size_t size = 0;
    char *bytes = path ? read_file_text(path, &size) : NULL;
    free(path);
    if (!bytes || size < sizeof(hybrid_runtime)) { free(bytes); return false; }
    hybrid_runtime state;
    memcpy(&state, bytes, sizeof(state));
    size_t expected = sizeof(state) + (size_t)state.request_bytes + state.last_read_bytes;
    for (int i = 0; i < QW3_AGENT_SOURCE_READ_TRACKED; i++) expected += state.source[i].bytes;
    if (state.magic != HYBRID_MAGIC || state.version != 2 || size != expected || state.transcript_tokens != a->transcript.len ||
        size > HYBRID_MAX_FRAME || state.turn.round < 0 ||
        state.turn.tool_history.n_items < 0 || state.turn.tool_history.n_items > QW3_AGENT_TOOL_HISTORY_MAX) {
        free(bytes); return false;
    }
    size_t pos = sizeof(state);
    free(a->turn_request);
    a->turn_request = strndup(bytes + pos, state.request_bytes); pos += state.request_bytes;
    free(a->last_read_path);
    a->last_read_path = state.last_read_bytes ? strndup(bytes + pos, state.last_read_bytes) : NULL;
    pos += state.last_read_bytes;
    agent_reset_source_read_budget(a);
    for (int i = 0; i < QW3_AGENT_SOURCE_READ_TRACKED; i++) {
        a->source_reads[i].chunks = state.source[i].chunks;
        a->source_reads[i].lines = state.source[i].lines;
        a->source_reads[i].path = state.source[i].bytes ? strndup(bytes + pos, state.source[i].bytes) : NULL;
        pos += state.source[i].bytes;
    }
    a->turn = state.turn; a->cfg.sample = state.sample;
    a->edit_verification_pending = state.pending;
    a->compact_retry_after_tokens = state.compact_retry;
    a->last_read_next = state.last_read_next;
    a->hybrid_generated = state.generated;
    free(bytes);
    return a->turn_request != NULL;
}

static bool agent_hybrid_checkpoint(agent_state *a) {
    if (!a->cfg.hybrid_job || a->cfg.hybrid_fd < 0) return true;
    /* On restart resume after this round; never replay its saved tool call. */
    a->turn.round++;
    double started = agent_now_sec();
    bool ok = store_save(a, a->cfg.conversation) == 0 && hybrid_save_runtime(a);
    agent_trace_begin(a, "hybrid_checkpoint");
    agent_trace_i64(a, "tokens", a->transcript.len);
    agent_trace_bool(a, "ok", ok);
    agent_trace_f64(a, "elapsed_ms", (agent_now_sec() - started) * 1000);
    agent_trace_end(a);
    a->turn.round--;
    return ok;
}

static bool hybrid_journal(agent_state *a, const tool_call *call, const char *stage, const char *out) {
    if (!a->cfg.hybrid_job) return true;
    char *path = path_join(a->cfg.hybrid_job, a->cfg.hybrid_dense ? "dense-tools.jsonl" : "moe-tools.jsonl");
    int fd = path ? open(path, O_WRONLY | O_CREAT | O_APPEND, 0600) : -1;
    free(path);
    if (fd < 0) return false;
    FILE *fp = fdopen(fd, "a");
    if (!fp) { close(fd); return false; }
    fprintf(fp, "{\"stage\":"); agent_trace_json_string(fp, stage);
    fprintf(fp, ",\"tool\":"); agent_trace_json_string(fp, call->name);
    fprintf(fp, ",\"round\":%d,\"pending\":%s,\"params\":{", a->turn.round,
            a->edit_verification_pending ? "true" : "false");
    bool first = true;
    for (int i = 0; i < call->n_params; i++) {
        /* Keep commands/paths, not a second copy of entire written files. */
        if (!strcmp(call->params[i].name, "content") || !strcmp(call->params[i].name, "new") ||
            !strcmp(call->params[i].name, "old")) continue;
        if (!first) fputc(',', fp);
        first = false;
        agent_trace_json_string(fp, call->params[i].name); fputc(':', fp);
        agent_trace_json_string(fp, call->params[i].value);
    }
    fprintf(fp, "},\"output\":"); agent_trace_json_string(fp, out ? out : "");
    fprintf(fp, "}\n");
    bool ok = fflush(fp) == 0 && fsync(fd) == 0;
    if (fclose(fp) != 0) ok = false;
    return ok;
}

static char *hybrid_normalize_command(const char *text) {
    strbuf normalized = {0};
    for (const unsigned char *p = (const unsigned char *)text; p && *p; p++)
        if (!isspace(*p)) sb_append_n(&normalized, (const char *)p, 1);
    return normalized.p ? normalized.p : agent_strdup("");
}

static bool hybrid_replay_blocked(agent_state *a, const tool_call *call) {
    if (strcmp(call->name, "bash")) return false;
    char *path = store_path_ext(a, a->cfg.conversation, ".hybrid-replay-blocks");
    char *blocks = path ? read_file_text(path, NULL) : NULL; free(path);
    const char *cmd = tool_param_value(call, "cmd");
    if (!cmd) cmd = tool_param_value(call, "command");
    char *normalized = hybrid_normalize_command(cmd); bool blocked = false;
    char *line = blocks;
    while (line && *line) {
        char *end = strchr(line, '\n'); if (end) *end = '\0';
        if (*line && strstr(normalized, line)) blocked = true;
        if (!end) break; line = end + 1;
    }
    free(normalized); free(blocks); return blocked;
}

static bool agent_hybrid_before_tool(agent_state *a, const tool_call *call) {
    if (!a->cfg.hybrid_job || a->cfg.hybrid_fd < 0) return true;
    if (hybrid_replay_blocked(a, call)) {
        agent_statusf(a, "agent: blocked replay of a recorded shell operation; inspect files and run a read-only check\n");
        return false;
    }
    bool mutating = !strcmp(call->name, "write") || !strcmp(call->name, "edit") || !strcmp(call->name, "bash");
    bool previous = a->edit_verification_pending;
    bool ok = true;
    if (mutating) {
        a->edit_verification_pending = true;
        ok = agent_hybrid_checkpoint(a);
    }
    if (ok) ok = hybrid_journal(a, call, "started", NULL);
    if (ok && !strcmp(call->name, "bash")) {
        const char *cmd = tool_param_value(call, "cmd");
        if (!cmd) cmd = tool_param_value(call, "command");
        char *normalized = hybrid_normalize_command(cmd);
        char *path = path_join(a->cfg.hybrid_job, a->cfg.hybrid_dense ? "dense-commands.txt" : "moe-commands.txt");
        int fd = path ? open(path, O_WRONLY | O_CREAT | O_APPEND, 0600) : -1;
        ok = fd >= 0 && hybrid_io(fd, normalized, strlen(normalized), true) && hybrid_io(fd, "\n", 1, true) && fsync(fd) == 0;
        if (fd >= 0) close(fd); free(path); free(normalized);
    }
    /* Restore successful read-only shell behavior; its on-disk in-flight state
     * and journal remain pending until the completed result is committed. */
    if (!strcmp(call->name, "bash")) a->edit_verification_pending = previous;
    return ok;
}

static void agent_hybrid_after_tool(agent_state *a, const tool_call *call, const char *out) {
    if (a->cfg.hybrid_job && !strcmp(call->name, "bash") && agent_should_interrupt(a))
        a->edit_verification_pending = true;
    if (!hybrid_journal(a, call, "completed", out)) {
        a->edit_verification_pending = true;
        agent_statusf(a, "agent: tool journal failed; verification remains pending\n");
    }
}

static bool hybrid_append_result(agent_state *a, const char *tool, const char *result) {
    char *text = native_tool_response_text(tool, result);
    char error[160] = {0};
    bool ok = agent_ensure_message_room(a, "user", text, 128, "delegation result", error, sizeof(error)) &&
              agent_append_message(a, AGENT_MESSAGE_TOOL, "user", text);
    if (!ok) agent_statusf(a, "agent: cannot append delegation result: %s\n", error);
    free(text);
    return ok;
}

static bool agent_hybrid_tool_budget(agent_state *a) {
    if (!a->cfg.hybrid_dense) return true;
    if (a->turn.dense_tool_rounds >= a->cfg.expert_rounds) {
        agent_statusf(a, "agent: expert tool-round budget exhausted; task incomplete\n");
        return false;
    }
    a->turn.dense_tool_rounds++;
    return true;
}

static bool hybrid_routine_survey(const char *request) {
    char *lower = agent_strdup(request ? request : ""); if (!lower) return false;
    for (char *p = lower; *p; p++) *p = (char)tolower((unsigned char)*p);
    const char *routine[] = {"analizza il progetto", "analizza questo progetto", "analyze the project",
                            "analizza la cartella", "struttura del progetto", "project structure", "explore the project"};
    const char *hard[] = {"bug", "errore", "corrett", "correct", "diagnos", "failed", "fallit", "deadlock",
                         "race", "progetta", "design decision", "decision", "architettural", "architectural"};
    bool survey = false;
    for (size_t i=0;i<sizeof(routine)/sizeof(routine[0]);i++) if (strstr(lower,routine[i])) survey = true;
    for (size_t i=0;i<sizeof(hard)/sizeof(hard[0]);i++) if (strstr(lower,hard[i])) survey = false;
    free(lower); return survey;
}

static const char *hybrid_delegation_error(agent_state *a, const tool_call_list *calls, const tool_call *call) {
    const char *error = NULL;
    if (a->cfg.hybrid_fd < 0 || !a->cfg.hybrid_job || a->cfg.hybrid_dense || !a->cfg.expert_model)
        error = "Delegation unavailable; recursive delegation is forbidden.";
    else if (a->cfg.expert_mode != AGENT_EXPERT_AUTO)
        error = "Automatic delegation is disabled. Continue with the current model.";
    else if (!a->cfg.tools_enabled) error = "Tools are disabled.";
    else if (!a->hybrid_forced && hybrid_routine_survey(a->turn_request))
        error = "Routine project exploration belongs to the primary MoE. Read/search the project yourself; delegation is for a concrete difficult diagnosis, failed verification, correctness review or architecture decision.";
    else if (a->turn.delegated) error = "Only one expert delegation is allowed per user turn.";
    else if (calls->n_calls != 1) error = "delegate_dense must be the only tool call in this batch; nothing in this batch was executed.";
    const char *task = tool_param_value(call, "task");
    const char *reason = tool_param_value(call, "reason");
    const char *context = tool_param_value(call, "context");
    const char *criteria = tool_param_value(call, "criteria");
    if (!error && (!task || !*task || !reason || !*reason || !criteria || !*criteria || !context))
        error = "delegate_dense requires task, reason, context and criteria strings; task/reason/criteria must be nonempty.";
    return error;
}

static int agent_hybrid_pause(agent_state *a, const tool_call_list *calls) {
    const tool_call *call = NULL;
    for (int i = 0; i < calls->n_calls; i++)
        if (!strcmp(calls->calls[i].name, "delegate_dense")) call = &calls->calls[i];
    if (!call) return 0;
    const char *error = hybrid_delegation_error(a, calls, call);
    const char *task = tool_param_value(call, "task");
    const char *reason = tool_param_value(call, "reason");
    const char *context = tool_param_value(call, "context");
    const char *criteria = tool_param_value(call, "criteria");
    if (error) { hybrid_append_result(a, "delegate_dense", error); return 1; }
    strbuf request = {0};
    sb_append(&request, "Delegated task. Complete it with the available tools. Do not delegate.\n"
              "Return solution, changed files, verification commands/results and remaining uncertainties.\n\nOriginal user request and constraints (verbatim):\n");
    sb_append(&request, a->turn_request ? a->turn_request : "");
    sb_printf(&request, "\n\nTask:\n%s\n\nDelegation reason:\n%s\n\nCompletion criteria:\n%s\n",
              task, reason, criteria);
    char *evidence_path = path_join(a->cfg.hybrid_job, "evidence.txt");
    sb_printf(&request, "\nFull collected evidence, attempts and source paths: %s\n"
              "Read that file when an excerpt below omits relevant details.\n", evidence_path ? evidence_path : "unavailable");
    size_t essential = request.len;
    sb_printf(&request, "\nCollected evidence, attempts and file paths:\n%s\n",
              context ? context : "No additional evidence provided; investigate using tools.");
    sb_append(&request, "\nSupplementary recent evidence (may be abbreviated; use the source paths to read more):\n");
    size_t base = request.len;
    for (int i = a->messages.len - 2; i >= 0 && request.len - base < 8000; i--) {
        agent_message *m = &a->messages.items[i];
        if (m->kind != AGENT_MESSAGE_TOOL) continue;
        size_t keep = strlen(m->content) < 2000 ? strlen(m->content) : 2000;
        while (keep && ((unsigned char)m->content[keep] & 0xc0) == 0x80) keep--;
        sb_append_n(&request, m->content, keep);
        sb_append(&request, "\n[excerpt; source paths/commands above]\n");
    }
    char *path = path_join(a->cfg.hybrid_job, "request.txt");
    char *base_path = path_join(a->cfg.hybrid_job, "request-base.txt");
    char *essential_path = path_join(a->cfg.hybrid_job, "request-essential.txt");
    bool ok = path && base_path && evidence_path && essential_path &&
              hybrid_atomic(evidence_path, context ? context : "", context ? strlen(context) : 0) &&
              hybrid_atomic(essential_path, request.p, essential) &&
              hybrid_atomic(base_path, request.p, base) &&
              hybrid_atomic(path, request.p, request.len);
    free(path); free(base_path); free(evidence_path); free(essential_path); sb_free(&request);
    a->turn.delegated = true;
    if (!ok || !agent_hybrid_checkpoint(a)) {
        hybrid_append_result(a, "delegate_dense", "Delegation failed to persist; expert was not started.");
        return 1;
    }
    agent_statusf(a, "agent: delegating to dense: %s\n", reason);
    hybrid_send(a->cfg.hybrid_fd, H_DELEGATE, reason, strlen(reason));
    return 2;
}

static void agent_hybrid_generation(agent_state *a) {
    if (a->cfg.hybrid_dense && a->cfg.hybrid_fd >= 0 && !(a->hybrid_generated % 32))
        hybrid_send(a->cfg.hybrid_fd, H_BUDGET, &a->hybrid_generated, sizeof(int));
}

static void hybrid_worker_output(void *ud, const char *text, size_t bytes) {
    agent_state *a = ud;
    if (!hybrid_send(a->cfg.hybrid_fd, H_OUTPUT, text, bytes)) a->hybrid_cancelled = true;
}
static void hybrid_worker_status(void *ud, const char *text, size_t bytes) {
    agent_state *a = ud;
    if (!hybrid_send(a->cfg.hybrid_fd, H_STATUS, text, bytes)) a->hybrid_cancelled = true;
}
static void hybrid_worker_activity(void *ud, const char *phase, const char *detail) {
    agent_state *a = ud;
    char text[128];
    snprintf(text, sizeof(text), "%s%s%s", phase, detail ? ":" : "", detail ? detail : "");
    hybrid_send(a->cfg.hybrid_fd, H_ACTIVITY, text, strlen(text));
}
static void hybrid_worker_progress(void *ud, const char *phase, int current, int total, double tps) {
    agent_state *a = ud;
    hybrid_progress p = {.current = current, .total = total, .tps = tps};
    snprintf(p.phase, sizeof(p.phase), "%s", phase);
    hybrid_send(a->cfg.hybrid_fd, H_PROGRESS, &p, sizeof(p));
}
static void hybrid_worker_timing(void *ud, const char *backend, int cached, int prompt,
                                 int prefill, double prefill_ms, double pp,
                                 int generated, double gen_ms, double tg) {
    agent_state *a = ud;
    hybrid_timing t = {.cached = cached, .prompt = prompt, .prefill = prefill,
        .prefill_ms = prefill_ms, .pp = pp, .generated = generated, .gen_ms = gen_ms, .tg = tg};
    snprintf(t.backend, sizeof(t.backend), "%s", backend);
    hybrid_send(a->cfg.hybrid_fd, H_TIMING, &t, sizeof(t));
}
static bool hybrid_worker_interrupt(void *ud) {
    agent_state *a = ud;
    if (a->hybrid_cancelled || (a->hybrid_deadline && agent_now_sec() >= a->hybrid_deadline)) return true;
    struct pollfd p = {.fd = a->cfg.hybrid_fd, .events = POLLIN};
    if (poll(&p, 1, 0) > 0) {
        uint32_t type; size_t bytes;
        char *data = hybrid_receive(p.fd, &type, &bytes);
        a->hybrid_cancelled = !data || type == H_CANCEL || type == H_STOP;
        free(data);
    }
    return a->hybrid_cancelled;
}

static bool hybrid_result_applied(const agent_state *a, const char *result) {
    const char *newline = strchr(result, '\n');
    if (!newline || strncmp(result, "Hybrid result ID: ", 18)) return false;
    char *marker = strndup(result, (size_t)(newline - result + 1));
    if (!marker) return false;
    bool applied = false;
    if (a->cfg.hybrid_job) {
        char *path = path_join(a->cfg.hybrid_job, "result-applied");
        char *committed = path ? read_file_text(path, NULL) : NULL; free(path);
        applied = committed && !strcmp(committed, marker); free(committed);
    }
    for (int i = 0; i < a->messages.len; i++)
        if (a->messages.items[i].kind == AGENT_MESSAGE_TOOL &&
            strstr(a->messages.items[i].content, marker)) applied = true;
    free(marker); return applied;
}

static int agent_hybrid_worker_loop(agent_state *a) {
    a->output_write = hybrid_worker_output; a->output_ud = a;
    a->status_write = hybrid_worker_status; a->status_ud = a;
    a->activity_update = hybrid_worker_activity; a->activity_ud = a;
    a->progress_update = hybrid_worker_progress; a->progress_ud = a;
    a->timing_update = hybrid_worker_timing; a->timing_ud = a;
    a->should_interrupt = hybrid_worker_interrupt; a->interrupt_ud = a;
    a->output_color = a->cfg.hybrid_color;
    if (!hybrid_load_runtime(a)) {
        a->turn.guard_retry_credits = 2; a->turn.empty_retry_credits = 2;
        a->turn.verification_guard_credits = 3;
        for (int i = a->messages.len - 1; i >= 0; i--)
            if (a->messages.items[i].kind == AGENT_MESSAGE_USER) {
                free(a->turn_request); a->turn_request = agent_strdup(a->messages.items[i].content); break;
            }
        if (!a->turn_request) a->turn_request = agent_strdup("Recover the interrupted task; inspect the durable tool journal.");
    }
    hybrid_done ready = {.tokens = a->transcript.len, .pending = a->edit_verification_pending};
    if (!hybrid_send(a->cfg.hybrid_fd, H_READY, &ready, sizeof(ready))) return 1;
    for (;;) {
        uint32_t type; size_t bytes;
        char *data = hybrid_receive(a->cfg.hybrid_fd, &type, &bytes);
        if (!data) break;
        if (type == H_STOP) { free(data); break; }
        a->hybrid_cancelled = false;
        int rc = -1;
        if (type == H_COMMAND) {
            char *message = NULL;
            if (!strcmp(data, "/new")) {
                a->edit_verification_pending = false;
                char *blocks = store_path_ext(a, a->cfg.conversation, ".hybrid-replay-blocks");
                if (blocks) unlink(blocks); free(blocks);
            }
            rc = handle_command(a, data, &message);
            if (message) { rc = run_agent_turn(a, message); free(message); }
            if (rc >= 0) { rc = 0; store_save(a, a->cfg.conversation); hybrid_save_runtime(a); }
        } else if (type == H_RUN && bytes >= sizeof(hybrid_run_options)) {
            hybrid_run_options options;
            memcpy(&options, data, sizeof(options));
            a->cfg.expert_mode = (agent_expert_mode)options.mode;
            if (options.job_bytes > PATH_MAX || sizeof(options) + options.job_bytes > bytes) { free(data); return 1; }
            free(a->hybrid_job_owned);
            a->hybrid_job_owned = strndup(data + sizeof(options), options.job_bytes);
            a->cfg.hybrid_job = a->hybrid_job_owned;
            const char *request = data + sizeof(options) + options.job_bytes;
            if (a->cfg.hybrid_dense) {
                a->hybrid_deadline = agent_now_sec() + a->cfg.expert_timeout;
                a->hybrid_generated = 0;
                /* Count the actual dense tokenizer/template including tool instructions. */
                qw3_tokens seed = {0};
                qw3_encode_chat_prompt(a->engine, a->cfg.system_prompt, request, a->cfg.think_mode, &seed);
                bool fits = seed.len <= 4096 && seed.len + 128 < a->cfg.ctx_size;
                qw3_tokens_free(&seed);
                if (!fits) {
                    char *base_path = path_join(a->cfg.hybrid_job, "request-base.txt");
                    char *base = read_file_text(base_path, NULL);
                    free(base_path);
                    if (base) {
                        qw3_encode_chat_prompt(a->engine, a->cfg.system_prompt, base, a->cfg.think_mode, &seed);
                        fits = seed.len <= 4096 && seed.len + 128 < a->cfg.ctx_size;
                        qw3_tokens_free(&seed);
                        if (fits) rc = run_agent_turn(a, base);
                        free(base);
                    }
                    if (!fits) {
                        char *essential_path = path_join(a->cfg.hybrid_job, "request-essential.txt");
                        char *essential = read_file_text(essential_path, NULL);
                        free(essential_path);
                        if (essential) {
                            qw3_encode_chat_prompt(a->engine, a->cfg.system_prompt, essential, a->cfg.think_mode, &seed);
                            fits = seed.len <= 4096 && seed.len + 128 < a->cfg.ctx_size;
                            qw3_tokens_free(&seed);
                            if (fits) {
                                /* Keep the user request/constraints intact. Abbreviate only
                                 * supplemental evidence, retaining its complete file path. */
                                char *evidence_path = path_join(a->cfg.hybrid_job, "evidence.txt");
                                char *evidence = read_file_text(evidence_path, NULL);
                                free(evidence_path);
                                size_t lo = 0, hi = evidence ? strlen(evidence) : 0;
                                while (lo < hi) {
                                    size_t mid = lo + (hi - lo + 1) / 2;
                                    size_t keep = mid;
                                    while (keep && ((unsigned char)evidence[keep] & 0xc0) == 0x80) keep--;
                                    strbuf candidate = {0}; sb_append(&candidate, essential);
                                    sb_append(&candidate, "\nAbbreviated evidence:\n");
                                    sb_append_n(&candidate, evidence, keep);
                                    sb_append(&candidate, "\n[Excerpt; read the full evidence file above.]\n");
                                    qw3_encode_chat_prompt(a->engine, a->cfg.system_prompt, candidate.p, a->cfg.think_mode, &seed);
                                    bool candidate_fits = seed.len <= 4096 && seed.len + 128 < a->cfg.ctx_size;
                                    qw3_tokens_free(&seed); sb_free(&candidate);
                                    if (candidate_fits) lo = mid; else hi = mid - 1;
                                }
                                strbuf bounded = {0}; sb_append(&bounded, essential);
                                if (lo) {
                                    while (lo && ((unsigned char)evidence[lo] & 0xc0) == 0x80) lo--;
                                    sb_append(&bounded, "\nAbbreviated evidence:\n");
                                    sb_append_n(&bounded, evidence, lo);
                                    sb_append(&bounded, "\n[Excerpt; read the full evidence file above.]\n");
                                }
                                const char *notice = "agent: supplemental expert evidence abbreviated; full evidence retained on disk\n";
                                hybrid_worker_status(a, notice, strlen(notice));
                                rc = run_agent_turn(a, bounded.p);
                                sb_free(&bounded); free(evidence);
                            }
                            free(essential);
                        }
                    }
                    if (!fits) {
                        const char *error = "agent: expert packet exceeds 4096 dense tokens; request/constraints were preserved, not truncated\n";
                        hybrid_worker_status(a, error, strlen(error));
                    }
                } else rc = run_agent_turn(a, request);
            } else if (options.force) {
                /* Forced delegation still creates one proper user turn and tool call. */
                memset(&a->turn, 0, sizeof(a->turn));
                a->turn.guard_retry_credits = 2; a->turn.empty_retry_credits = 2;
                a->turn.verification_guard_credits = 3;
                free(a->turn_request); a->turn_request = agent_strdup(request);
                agent_reset_source_read_budget(a);
                agent_note_user_message(a, request);
                if (agent_append_message(a, AGENT_MESSAGE_USER, "user", request)) {
                    tool_call_list call = {.n_calls = 1};
                    snprintf(call.calls[0].name, sizeof(call.calls[0].name), "delegate_dense");
                    const char *names[] = {"task", "reason", "criteria", "context"};
                    const char *values[] = {"Complete the original user request and constraints.", "Explicit user request", "Complete the user request; verify any changes and report results.", "Inspect source paths and tool journals when needed."};
                    call.calls[0].n_params = 4;
                    for (int i = 0; i < 4; i++) {
                        snprintf(call.calls[0].params[i].name, sizeof(call.calls[0].params[i].name), "%s", names[i]);
                        call.calls[0].params[i].value = (char *)values[i];
                    }
                    char *markup = native_tool_call_text(&call);
                    qw3_tokens tokens = {0};
                    int start = a->transcript.len;
                    qw3_chat_append_assistant_prefix(a->engine, &a->transcript, QW3_THINK_NONE);
                    qw3_tokenize_text(a->engine, markup, &tokens);
                    append_generated_assistant(a, &tokens, start, markup);
                    qw3_tokens_free(&tokens); free(markup);
                    agent_expert_mode mode = a->cfg.expert_mode;
                    a->cfg.expert_mode = AGENT_EXPERT_AUTO; a->hybrid_forced = true;
                    rc = agent_hybrid_pause(a, &call) == 2 ? 2 : -1;
                    a->cfg.expert_mode = mode; a->hybrid_forced = false;
                }
            } else {
                char control[160];
                snprintf(control, sizeof(control), "Internal control: expert mode is %s. %s", hybrid_mode_name(a->cfg.expert_mode),
                         a->cfg.expert_mode == AGENT_EXPERT_AUTO ? "Delegate only when the task warrants it." : "Do not call delegate_dense.");
                if (a->cfg.expert_mode != AGENT_EXPERT_OFF && a->cfg.tools_enabled)
                    agent_append_message(a, AGENT_MESSAGE_CONTROL, "user", control);
                rc = run_agent_turn(a, request);
            }
        } else if (type == H_CONTINUE && bytes >= sizeof(hybrid_run_options)) {
            hybrid_run_options options; memcpy(&options, data, sizeof(options));
            a->edit_verification_pending |= options.mode != 0;
            /* An applied result is committed in the same checkpoint/runtime pair. */
            const char *result = data + sizeof(options);
            bool applied = hybrid_result_applied(a, result);
            if (applied || hybrid_append_result(a, "delegate_dense", result)) {
                /* Commit the result before any new model/tool work. Keep the old
                 * KV untouched so the measured continuation performs its prefill.
                 * The small transcript/runtime bundle is atomic and sufficient
                 * for a text rebuild if the old KV no longer matches. */
                a->hybrid_result_commit = true;
                bool committed = store_save(a, a->cfg.conversation) == 0 && hybrid_save_runtime(a);
                a->hybrid_result_commit = false;
                const char *end = strchr(result, '\n');
                char *marker_path = path_join(a->cfg.hybrid_job, "result-applied");
                committed = committed && end && marker_path &&
                    hybrid_atomic(marker_path, result, (size_t)(end-result+1));
                free(marker_path);
                if (committed) {
                    a->hybrid_resume = options.force && !applied;
                    rc = a->hybrid_resume ? run_agent_turn(a, a->turn_request) : 0;
                }
            }
        }
        free(data);
        if (rc == 2) continue;
        bool cancelled = agent_should_interrupt(a);
        if (cancelled) rc = -2;
        bool saved = store_save(a, a->cfg.conversation) == 0 && hybrid_save_runtime(a);
        if (!saved) rc = -1;
        strbuf result = {0};
        if (a->cfg.hybrid_dense) {
            char *journal_path = path_join(a->cfg.hybrid_job, "dense-tools.jsonl");
            char *journal = read_file_text(journal_path, NULL);
            sb_printf(&result, "Expert status: %s\nVerification pending: %s\nFull evidence: %s\n",
                      rc == 0 ? "completed" : cancelled ? "interrupted/timeout" : "failed/incomplete",
                      a->edit_verification_pending ? "yes" : "no", journal_path);
            sb_printf(&result, "Generated tokens used: %d/%d\nTool rounds used: %d/%d\n",
                      a->hybrid_generated, a->cfg.expert_tokens, a->turn.dense_tool_rounds, a->cfg.expert_rounds);
            /* Put the solution before verbose read results: the parent may
             * abbreviate the return packet, with the full audit still on disk. */
            for (int i = a->messages.len - 1; i >= 0; i--) if (a->messages.items[i].kind == AGENT_MESSAGE_ASSISTANT) {
                char *answer = agent_compact_without_thinking(a->messages.items[i].content, NULL);
                sb_append(&result, rc == 0 ? "\nExpert answer:\n" : "\nLast saved expert message (task incomplete):\n");
                sb_append(&result, answer); free(answer); break;
            }
            if (journal) {
                sb_append(&result, "Tool audit (authoritative):\n");
                size_t keep = strlen(journal) < 64000 ? strlen(journal) : 64000;
                while (keep && ((unsigned char)journal[keep] & 0xc0) == 0x80) keep--;
                sb_append_n(&result, journal, keep);
                if (journal[keep]) sb_append(&result, "\n[audit excerpt; full evidence at the path above]\n");
            }
            free(journal); free(journal_path);
        }
        hybrid_done done = {.rc = rc, .tokens = a->transcript.len,
            .pending = a->edit_verification_pending, .generated = a->hybrid_generated};
        strbuf payload = {0}; sb_append_n(&payload, (char *)&done, sizeof(done));
        sb_append_n(&payload, result.p ? result.p : "", result.len);
        hybrid_send(a->cfg.hybrid_fd, H_DONE, payload.p, payload.len);
        sb_free(&payload); sb_free(&result);
    }
    return 0;
}

static void hybrid_trace(agent_state *a, const char *event, bool dense, pid_t pid,
                          double elapsed, int rc) {
    agent_trace_begin(a, event);
    agent_trace_string(a, "role", dense ? "dense" : "moe");
    agent_trace_i64(a, "pid", pid);
    agent_trace_f64(a, "elapsed_ms", elapsed * 1000);
    agent_trace_i64(a, "rc", rc);
    hybrid_supervisor *h = a->hybrid_supervisor;
    agent_trace_string(a, "job", h && h->job ? h->job : "");
    agent_trace_end(a);
}

static void hybrid_logs(agent_state *a, hybrid_process *process) {
    if (process->logs < 0) return;
    char buffer[8192];
    ssize_t n;
    while ((n = read(process->logs, buffer, sizeof(buffer))) > 0)
        agent_status_write(a, buffer, (size_t)n);
}

static void hybrid_close_process(agent_state *a, bool graceful) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    hybrid_process *p = &h->process;
    if (p->pid <= 0) return;
    double start = agent_now_sec();
    if (graceful) hybrid_send(p->fd, H_STOP, NULL, 0);
    else kill(-p->pid, SIGTERM);
    int status = 0;
    double deadline = start + 5.0;
    while (waitpid(p->pid, &status, WNOHANG) == 0) {
        hybrid_logs(a, p);
        if (agent_now_sec() >= deadline) {
            kill(-p->pid, SIGKILL);
            while (waitpid(p->pid, &status, 0) < 0 && errno == EINTR) {}
            break;
        }
        struct pollfd fds[2] = {{p->fd, POLLIN, 0}, {p->logs, POLLIN, 0}};
        poll(fds, 2, 50);
        if (fds[0].revents & POLLIN) {
            uint32_t type; size_t bytes;
            char *data = hybrid_receive(p->fd, &type, &bytes); free(data);
        }
    }
    hybrid_logs(a, p);
    hybrid_trace(a, "hybrid_process_exit", p->dense, p->pid, agent_now_sec() - start, status);
    close(p->fd); close(p->logs);
    *p = (hybrid_process){.fd = -1, .logs = -1};
}

static bool hybrid_start(agent_state *a, bool dense) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    if (h->process.pid > 0) return false;
    int logs[2], sockets[2];
    if (pipe(logs) != 0) return false;
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        close(logs[0]); close(logs[1]); return false;
    }
#ifdef SO_NOSIGPIPE
    int yes = 1;
    setsockopt(sockets[0], SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
    setsockopt(sockets[1], SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    for (int i = 0; i < 2; i++) {
        fcntl(logs[i], F_SETFD, FD_CLOEXEC);
        fcntl(sockets[i], F_SETFD, FD_CLOEXEC);
    }
    char ctx[24], tokens[24], rounds[24], timeout[24], total[24], ngl[24];
    char temp[32], topk[24], topp[32], minp[32], repeat[32], lastn[24], seed[32];
    snprintf(ctx, sizeof(ctx), "%d", dense ? a->cfg.expert_ctx : a->cfg.ctx_size);
    snprintf(tokens, sizeof(tokens), "%d", dense ? a->cfg.expert_tokens : a->cfg.n_predict);
    snprintf(rounds, sizeof(rounds), "%d", dense ? a->cfg.expert_rounds : a->cfg.max_tool_rounds);
    snprintf(timeout, sizeof(timeout), "%d", a->cfg.expert_timeout);
    snprintf(total, sizeof(total), "%d", a->cfg.expert_tokens);
    snprintf(ngl, sizeof(ngl), "%d", a->cfg.metal_n_gpu_layers);
    snprintf(temp, sizeof(temp), "%.9g", a->cfg.sample.temperature);
    snprintf(topk, sizeof(topk), "%d", a->cfg.sample.sample_top_k);
    snprintf(topp, sizeof(topp), "%.9g", a->cfg.sample.top_p);
    snprintf(minp, sizeof(minp), "%.9g", a->cfg.sample.min_p);
    snprintf(repeat, sizeof(repeat), "%.9g", a->cfg.sample.repeat_penalty);
    snprintf(lastn, sizeof(lastn), "%d", a->cfg.sample.repeat_last_n);
    snprintf(seed, sizeof(seed), "%llu", (unsigned long long)a->cfg.sample.rng);
    char *store = dense ? path_join(h->job, "dense-store") : agent_strdup(h->primary_store);
    char *trace = path_join(dense ? h->job : h->primary_store, dense ? "dense.trace.jsonl" : "moe.trace.jsonl");
    if (!store || !trace || ensure_private_dir(store) != 0) {
        free(store); free(trace); for (int i = 0; i < 2; i++) { close(logs[i]); close(sockets[i]); } return false;
    }
    char *argv[80]; int n = 0;
#define ARG(S) argv[n++] = (char *)(S)
    ARG(h->executable); ARG("--hybrid-worker-fd"); ARG("3");
    ARG("--hybrid-lock"); ARG(h->lock);
    ARG("-m"); ARG(dense ? h->dense_model : h->primary_model);
    ARG(a->cfg.backend == QW3_BACKEND_METAL ? "--metal" : "--cpu");
    ARG("--ctx"); ARG(ctx); ARG("-n"); ARG(tokens);
    ARG("--max-tool-rounds"); ARG(rounds);
    ARG("--store-dir"); ARG(store); ARG("--conversation"); ARG(dense ? "expert" : a->cfg.conversation);
    ARG("--trace"); ARG(trace);
    ARG("--temp"); ARG(temp); ARG("--sample-top-k"); ARG(topk);
    ARG("--top-p"); ARG(topp); ARG("--min-p"); ARG(minp);
    ARG("--repeat-penalty"); ARG(repeat); ARG("--repeat-last-n"); ARG(lastn);
    ARG("--seed"); ARG(seed);
    if (!a->cfg.prefix_cache) ARG("--no-prefix-cache");
    if (!a->cfg.tools_enabled) ARG("--no-tools");
    if (a->cfg.read_only) ARG("--read-only");
    if (a->cfg.compact_tools) ARG("--compact-tools");
    if (a->cfg.user_system) { ARG("-sys"); ARG(a->cfg.user_system); }
    if (h->job && !dense) { ARG("--hybrid-job"); ARG(h->job); }
    if (a->output_color && !getenv("NO_COLOR")) ARG("--hybrid-color");
    if (dense) {
        ARG("--hybrid-dense"); ARG("--hybrid-job"); ARG(h->job);
        ARG("--expert-timeout"); ARG(timeout); ARG("--expert-max-tokens"); ARG(total);
        if (a->cfg.backend == QW3_BACKEND_METAL) { ARG(a->cfg.expert_kv_f32 ? "--kv-f32" : "--kv-f16"); }
        ARG("--expert-think"); ARG(a->cfg.expert_think == QW3_THINK_NONE ? "none" : a->cfg.expert_think == QW3_THINK_HIGH ? "high" : "on");
    } else {
        ARG("--expert-model"); ARG(h->dense_model);
        ARG("--expert-mode"); ARG(hybrid_mode_name(a->cfg.expert_mode));
        if (a->cfg.metal_n_gpu_layers_set) { ARG("--ngl"); ARG(ngl); }
        if (a->cfg.think_mode == QW3_THINK_NONE) ARG("--nothink");
    }
    argv[n] = NULL;
#undef ARG
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, logs[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, logs[1], STDERR_FILENO);
    posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
    for (int i = 0; i < 2; i++) {
        if (logs[i] > 3) posix_spawn_file_actions_addclose(&actions, logs[i]);
        if (sockets[i] > 3) posix_spawn_file_actions_addclose(&actions, sockets[i]);
    }
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    pid_t pid = 0;
    int rc = posix_spawn(&pid, h->executable, &actions, &attr, argv, environ);
    posix_spawnattr_destroy(&attr); posix_spawn_file_actions_destroy(&actions);
    close(logs[1]); close(sockets[1]); free(store); free(trace);
    if (rc != 0) { close(logs[0]); close(sockets[0]); return false; }
    agent_set_nonblock(logs[0]); agent_set_nonblock(sockets[0]);
    h->process = (hybrid_process){.pid = pid, .fd = sockets[0], .logs = logs[0], .dense = dense};
    h->started = agent_now_sec(); h->generated = 0;
    snprintf(h->activity, sizeof(h->activity), "loading");
    agent_set_activity(a, dense ? "dense" : "moe", "loading");
    hybrid_trace(a, "hybrid_process_start", dense, pid, 0, 0);
    return true;
}

/* Waits in short polls so the editor's worker remains interruptible. */
static int hybrid_wait(agent_state *a, bool ready, char **result, bool *pending) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    hybrid_process *process = &h->process;
    bool cancellation_sent = false;
    double cancellation_at = 0;
    for (;;) {
        if (process->dense) {
            char detail[96];
            int left = a->cfg.expert_timeout - (int)(agent_now_sec() - h->started);
            snprintf(detail, sizeof(detail), "%.24s %ds/%dt left", h->activity,
                     left > 0 ? left : 0, a->cfg.expert_tokens - h->generated);
            agent_set_activity(a, "dense", detail);
        }
        bool interrupt = agent_should_interrupt(a);
        bool timeout = process->dense && agent_now_sec() - h->started >= a->cfg.expert_timeout;
        if ((interrupt || timeout) && !cancellation_sent) {
            hybrid_send(process->fd, H_CANCEL, NULL, 0);
            cancellation_sent = true; cancellation_at = agent_now_sec();
            hybrid_trace(a, timeout ? "hybrid_timeout" : "hybrid_interrupt", process->dense, process->pid, cancellation_at - h->started, -2);
        }
        if ((cancellation_sent && agent_now_sec() - cancellation_at > 5) ||
            (ready && agent_now_sec() - h->started > 120)) return -2;
        struct pollfd fds[2] = {{process->fd, POLLIN, 0}, {process->logs, POLLIN, 0}};
        int rc = poll(fds, 2, 100);
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0) return -1;
        hybrid_logs(a, process);
        if (fds[0].revents & (POLLIN | POLLHUP | POLLERR)) {
            uint32_t type; size_t bytes;
            char *data = hybrid_receive(process->fd, &type, &bytes);
            if (!data) return -1;
            if (type == H_OUTPUT) agent_output_write(a, data, bytes);
            else if (type == H_STATUS) agent_status_write(a, data, bytes);
            else if (type == H_BUDGET && bytes == sizeof(int)) {
                memcpy(&h->generated, data, sizeof(int));
            }
            else if (type == H_ACTIVITY) {
                snprintf(h->activity, sizeof(h->activity), "%s", data);
                agent_set_activity(a, process->dense ? "dense" : "moe", data);
            } else if (type == H_PROGRESS && bytes == sizeof(hybrid_progress)) {
                hybrid_progress p; memcpy(&p, data, sizeof(p));
                if (a->progress_update) a->progress_update(a->progress_ud, p.phase, p.current, p.total, p.tps);
            } else if (type == H_TIMING && bytes == sizeof(hybrid_timing)) {
                hybrid_timing t; memcpy(&t, data, sizeof(t));
                a->transcript.len = t.prompt;
                /* Generation count is supplied cumulatively by H_BUDGET. */
                if (a->timing_update) a->timing_update(a->timing_ud, t.backend, t.cached, t.prompt, t.prefill, t.prefill_ms, t.pp, t.generated, t.gen_ms, t.tg);
                agent_trace_begin(a, "hybrid_inference");
                agent_trace_string(a, "role", process->dense ? "dense" : "moe");
                agent_trace_i64(a, "prefill_tokens", t.prefill); agent_trace_f64(a, "prefill_ms", t.prefill_ms);
                agent_trace_i64(a, "generation_tokens", t.generated); agent_trace_f64(a, "generation_ms", t.gen_ms);
                agent_trace_end(a);
            } else if (type == H_DELEGATE && !ready) {
                agent_trace_begin(a, "hybrid_delegation");
                agent_trace_string(a, "reason", data);
                agent_trace_string(a, "mode", hybrid_mode_name(a->cfg.expert_mode));
                agent_trace_end(a);
                if (result) *result = agent_strdup(data);
                free(data); return 2;
            } else if ((type == H_READY && ready) || type == H_DONE) {
                if (bytes < sizeof(hybrid_done)) { free(data); return -1; }
                hybrid_done done; memcpy(&done, data, sizeof(done));
                a->transcript.len = done.tokens;
                if (pending) *pending = done.pending != 0;
                if (result) *result = strndup(data + sizeof(done), bytes - sizeof(done));
                hybrid_trace(a, ready ? "hybrid_ready" : "hybrid_completed", process->dense, process->pid, agent_now_sec() - h->started, done.rc);
                free(data); return cancellation_sent ? -2 : done.rc;
            } else if (type == H_ERROR) { agent_statusf(a, "agent: worker error: %s\n", data); free(data); return -1; }
            free(data);
        }
    }
}

static bool hybrid_phase(agent_state *a, int phase) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    char *path = path_join(h->job, "phase");
    char value[24]; int n = snprintf(value, sizeof(value), "%d\n", phase);
    bool ok = path && hybrid_atomic(path, value, (size_t)n);
    free(path);
    if (ok) h->phase = phase;
    return ok;
}

static bool hybrid_new_job(agent_state *a) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    free(h->job);
    h->job = path_join(h->root, "job-XXXXXX");
    if (!h->job || !mkdtemp(h->job)) return false;
    return hybrid_phase(a, H_PHASE_PRIMARY) && hybrid_atomic(h->active, h->job, strlen(h->job));
}

static bool hybrid_send_run(agent_state *a, const char *request, bool force) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    hybrid_run_options options = {.mode = a->cfg.expert_mode, .force = force, .job_bytes = (uint32_t)strlen(h->job)};
    strbuf data = {0};
    sb_append_n(&data, (char *)&options, sizeof(options));
    sb_append(&data, h->job); sb_append(&data, request);
    bool ok = hybrid_send(h->process.fd, H_RUN, data.p, data.len);
    sb_free(&data); return ok;
}

static int hybrid_return(agent_state *a, const char *result, bool pending, bool generate) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    if (!hybrid_start(a, false) || hybrid_wait(a, true, NULL, NULL) != 0) return -1;
    hybrid_run_options options = {.mode = pending, .force = generate};
    strbuf data = {0}; sb_append_n(&data, (char *)&options, sizeof(options));
    sb_printf(&data, "Hybrid result ID: %s\n", h->job);
    size_t result_len = strlen(result);
    size_t keep = result_len > 12000 ? 12000 : result_len;
    while (keep && ((unsigned char)result[keep] & 0xc0) == 0x80) keep--;
    sb_append_n(&data, result, keep);
    if (keep < result_len) sb_printf(&data, "\n[Result abbreviated; full result: %s/result.txt]\n", h->job);
    bool sent = hybrid_send(h->process.fd, H_CONTINUE, data.p, data.len);
    sb_free(&data);
    if (!sent) return -1;
    return hybrid_wait(a, false, NULL, NULL);
}

static void hybrid_finish(agent_state *a) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    if (hybrid_phase(a, H_PHASE_DONE)) {
        unlink(h->active);
        char *store = path_join(h->job, "dense-store");
        char *checkpoint = path_join(store, "expert.kvc");
        if (checkpoint) unlink(checkpoint);
        free(checkpoint); free(store);
    }
}

static bool hybrid_changes_possible(const char *job) {
    const char *names[] = {"dense-tools.jsonl", "moe-tools.jsonl"};
    bool possible = false;
    for (int i = 0; i < 2; i++) {
        char *path = path_join(job, names[i]); char *text = read_file_text(path, NULL); free(path);
        if (text && (strstr(text, "\"tool\":\"bash\"") || strstr(text, "\"tool\":\"write\"") ||
                     strstr(text, "\"tool\":\"edit\""))) possible = true;
        free(text);
    }
    return possible;
}

static bool hybrid_block_recorded_commands(agent_state *a) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    agent_state storage = {0}; storage.cfg = a->cfg; storage.cfg.store_dir = h->primary_store;
    char *target = store_path_ext(&storage, a->cfg.conversation, ".hybrid-replay-blocks");
    strbuf blocks = {0}; char *prior = target ? read_file_text(target, NULL) : NULL;
    if (prior) sb_append(&blocks, prior); free(prior);
    const char *names[] = {"dense-commands.txt", "moe-commands.txt"};
    for (int i=0;i<2;i++) {
        char *path = path_join(h->job, names[i]); char *text = path ? read_file_text(path, NULL) : NULL; free(path);
        if (text) sb_append(&blocks,text); free(text);
    }
    bool ok = target && hybrid_atomic(target, blocks.p ? blocks.p : "", blocks.len);
    free(target); sb_free(&blocks); return ok;
}

static int hybrid_recover(agent_state *a) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    if (!h->job || h->phase == H_PHASE_DONE) return 0;
    agent_statusf(a, "agent: recovering interrupted hybrid job %s; saved tools will not be replayed\n", h->job);
    char *path = path_join(h->job, "result.txt");
    char *result = read_file_text(path, NULL); free(path);
    if (!result) result = agent_strdup("Previous hybrid worker stopped unexpectedly. NEVER repeat previously recorded write/edit/shell operations, including with background wrappers. Inspect existing changes using read first and use read-only checks or tests; do not repeat the original task to verify it. Changes may exist and require inspection and relevant verification. Consult the job's moe-tools.jsonl and dense-tools.jsonl journals.");
    hybrid_close_process(a, true);
    if (!hybrid_block_recorded_commands(a)) { free(result); return -1; }
    char *pending_path = path_join(h->job, "pending");
    char *pending_text = pending_path ? read_file_text(pending_path, NULL) : NULL; free(pending_path);
    bool pending = pending_text && (*pending_text == '0' || *pending_text == '1') ?
                   *pending_text == '1' : hybrid_changes_possible(h->job);
    free(pending_text);
    int rc = hybrid_return(a, result, pending, false);
    free(result);
    if (rc == 0) hybrid_finish(a);
    return rc;
}

static int agent_hybrid_run(agent_state *a, const char *request) {
    hybrid_supervisor *h = a->hybrid_supervisor;
    if (!h) return -1;
    if (h->job && h->phase != H_PHASE_DONE && hybrid_recover(a) != 0) return -1;
    if (!strncmp(request, "/hybrid-command ", 16)) {
        if (!h->process.pid && (!hybrid_start(a, false) || hybrid_wait(a, true, NULL, NULL) != 0)) return -1;
        const char *command = request + 16;
        if (!hybrid_send(h->process.fd, H_COMMAND, command, strlen(command))) return -1;
        return hybrid_wait(a, false, NULL, NULL);
    }
    bool force = !strncmp(request, "/expert run ", 12);
    if (force) request += 12;
    if (force && (a->cfg.expert_mode == AGENT_EXPERT_OFF || !a->cfg.tools_enabled)) {
        agent_statusf(a, "agent: expert delegation is disabled\n"); return 0;
    }
    if (!hybrid_new_job(a)) return -1;
    char *request_path = path_join(h->job, "user-request.txt");
    bool persisted = request_path && hybrid_atomic(request_path, request, strlen(request));
    free(request_path); if (!persisted) return -1;
    if (!h->process.pid && (!hybrid_start(a, false) || hybrid_wait(a, true, NULL, NULL) != 0)) return -1;
    h->started = agent_now_sec();
    if (!hybrid_send_run(a, request, force)) return -1;
    char *reason = NULL;
    int rc = hybrid_wait(a, false, &reason, &h->pending);
    if (rc != 2) {
        free(reason);
        if (rc == 0) hybrid_finish(a);
        if (rc == -2) {
            hybrid_close_process(a, true);
            agent_interrupt_fn interrupt = a->should_interrupt; a->should_interrupt = NULL;
            int recovered = hybrid_recover(a); a->should_interrupt = interrupt;
            return recovered;
        }
        return rc;
    }
    agent_statusf(a, "agent: dense expert requested (%s), budget %ds / %d tokens\n", reason ? reason : "", a->cfg.expert_timeout, a->cfg.expert_tokens);
    free(reason);
    if (!hybrid_phase(a, H_PHASE_EXPERT)) return -1;
    hybrid_close_process(a, true);
    bool pending = false;
    char *result = NULL;
    if (hybrid_start(a, true) && hybrid_wait(a, true, NULL, NULL) == 0) {
        char *path = path_join(h->job, "request.txt");
        char *packet = read_file_text(path, NULL); free(path);
        if (packet && hybrid_send_run(a, packet, false)) rc = hybrid_wait(a, false, &result, &pending);
        else rc = -1;
        free(packet);
    } else rc = -1;
    hybrid_close_process(a, rc >= 0);
    if (!result || !*result) {
        free(result);
        result = agent_strdup("Dense expert unavailable, interrupted or failed. Do not claim completion. Inspect any existing changes and verify them. Saved tool journals are in this job directory.");
        pending = hybrid_changes_possible(h->job);
    }
    if (rc != 0) {
        pending |= hybrid_changes_possible(h->job);
        if (!hybrid_block_recorded_commands(a)) { free(result); return -1; }
    }
    char *result_path = path_join(h->job, "result.txt");
    char *pending_path = path_join(h->job, "pending");
    bool durable = hybrid_atomic(result_path, result, strlen(result)) &&
                   hybrid_atomic(pending_path, pending ? "1" : "0", 1) && hybrid_phase(a, H_PHASE_RETURN);
    free(result_path); free(pending_path);
    if (!durable) { free(result); return -1; }
    bool generate = !agent_should_interrupt(a);
    /* An interrupt cancels the expert, but must not prevent loading the saved MoE. */
    agent_interrupt_fn interrupt = a->should_interrupt;
    if (!generate) a->should_interrupt = NULL;
    rc = hybrid_return(a, result, pending, generate);
    a->should_interrupt = interrupt;
    free(result);
    if (rc == 0 || rc == -2) hybrid_finish(a);
    return rc == -2 ? 0 : rc;
}

static int agent_hybrid_command(agent_state *a, char *line, char **message) {
    *message = NULL;
    if (line[0] != '/') { *message = agent_strdup(line); return 1; }
    if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) return -1;
    if (!strcmp(line, "/help")) {
        interactive_help();
        agent_statusf(a, "  /expert run TASK       Force one dense delegation\n  /expert mode auto|manual|off\n  /expert status\n");
        return 0;
    }
    if (!strcmp(line, "/expert status")) {
        agent_statusf(a, "expert=%s mode=%s ctx=%d timeout=%ds tokens=%d rounds=%d\n",
                      a->cfg.expert_model, hybrid_mode_name(a->cfg.expert_mode), a->cfg.expert_ctx,
                      a->cfg.expert_timeout, a->cfg.expert_tokens, a->cfg.expert_rounds);
        return 0;
    }
    if (!strncmp(line, "/expert mode ", 13)) {
        const char *mode = line + 13;
        if (!strcmp(mode, "auto")) a->cfg.expert_mode = AGENT_EXPERT_AUTO;
        else if (!strcmp(mode, "manual")) a->cfg.expert_mode = AGENT_EXPERT_MANUAL;
        else if (!strcmp(mode, "off")) a->cfg.expert_mode = AGENT_EXPERT_OFF;
        else { agent_statusf(a, "agent: expert mode must be auto, manual or off\n"); return 0; }
        agent_statusf(a, "agent: expert mode=%s\n", mode); return 0;
    }
    if (!strncmp(line, "/expert run ", 12)) { *message = agent_strdup(line); return 1; }
    if (!strncmp(line, "/switch ", 8) || !strncmp(line, "/load ", 6)) {
        const char *name = line + (!strncmp(line, "/switch ", 8) ? 8 : 6);
        hybrid_supervisor *h = a->hybrid_supervisor;
        char *target = path_join(h->primary_store, name);
        strbuf file = {0}; sb_printf(&file, "%s.qw3a", target ? target : "");
        free(target);
        char *safe = sanitized_name(name);
        bool valid = safe && !strcmp(safe, name) && *name; free(safe);
        if (!valid || access(file.p, R_OK) != 0) {
            agent_statusf(a, "agent: cannot switch to %s\n", name); sb_free(&file); return 0;
        }
        sb_free(&file); hybrid_close_process(a, true);
        free(a->cfg.conversation); a->cfg.conversation = agent_strdup(name);
        uint64_t hash = agent_hash_update(1469598103934665603ull, h->primary_model, strlen(h->primary_model));
        struct stat st; if (stat(h->primary_model, &st) == 0) hash = agent_hash_update(hash, &st.st_size, sizeof(st.st_size));
        hash = agent_hash_update(hash, name, strlen(name));
        char active[64]; snprintf(active, sizeof(active), "active-%016llx", (unsigned long long)hash);
        free(h->active); h->active = path_join(h->root, active);
        free(h->job); h->job = read_file_text(h->active, NULL); h->phase = H_PHASE_PRIMARY;
        agent_statusf(a, "agent: switched to %s\n", name); return 0;
    }
    if (!strncmp(line, "/chdir", 6)) {
        char *path = line[6] == ' ' ? line + 7 : "";
        if (path[0]) {
            hybrid_close_process(a, true);
            handle_chdir_command(a, path);
        } else handle_chdir_command(a, "");
        return 0;
    }
    if (!strcmp(line, "/tools on")) a->cfg.tools_enabled = true;
    if (!strcmp(line, "/tools off")) a->cfg.tools_enabled = false;
    if (!strcmp(line, "/think")) a->cfg.think_mode = QW3_THINK_ON;
    if (!strcmp(line, "/nothink")) a->cfg.think_mode = QW3_THINK_NONE;
    /* A /read starts a normal supervised turn, so delegation and journaling work. */
    if (!strncmp(line, "/read ", 6)) {
        *message = read_file_text(line + 6, NULL);
        if (!*message) { agent_statusf(a, "agent: cannot read %s\n", line + 6); return 0; }
        return 1;
    }
    strbuf command = {0}; sb_append(&command, "/hybrid-command "); sb_append(&command, line);
    *message = command.p; return 1;
}

static bool hybrid_direct_interrupt(void *ud) { (void)ud; return g_agent_sigint != 0; }

static int agent_hybrid_main(agent_state *a, const char *executable) {
    hybrid_supervisor h = {.process = {.fd = -1, .logs = -1}};
    a->hybrid_supervisor = &h;
    h.executable = hybrid_absolute(executable);
    h.primary_model = hybrid_absolute(a->cfg.model_path);
    h.dense_model = hybrid_absolute(a->cfg.expert_model);
    h.root = path_join(a->cfg.store_dir, "hybrid");
    char *root = hybrid_absolute(h.root); free(h.root); h.root = root;
    if (!h.executable || !h.primary_model || !h.dense_model || !h.root || ensure_private_dir(h.root) != 0) return 1;
    uint64_t profile = agent_hash_update(1469598103934665603ull, h.primary_model, strlen(h.primary_model));
    struct stat st;
    if (stat(h.primary_model, &st) == 0) profile = agent_hash_update(profile, &st.st_size, sizeof(st.st_size));
    char name[64]; snprintf(name, sizeof(name), "moe-%016llx", (unsigned long long)profile);
    h.primary_store = path_join(h.root, name);
    if (!h.primary_store || ensure_private_dir(h.primary_store) != 0) return 1;
    if (!a->cfg.conversation) {
        char id[80]; snprintf(id, sizeof(id), "hybrid-%ld-%ld", (long)time(NULL), (long)getpid());
        a->cfg.conversation = agent_strdup(id);
    }
    uint64_t session = agent_hash_update(profile, a->cfg.conversation, strlen(a->cfg.conversation));
    snprintf(name, sizeof(name), "active-%016llx", (unsigned long long)session);
    h.active = path_join(h.root, name);
    h.lock = path_join(h.root, "engine.lock");
    char *previous = read_file_text(h.active, NULL);
    if (previous) {
        /* Active manifests can only refer to jobs below our private root. */
        if (!strncmp(previous, h.root, strlen(h.root)) && previous[strlen(h.root)] == '/') {
            h.job = previous;
            char *path = path_join(h.job, "phase"); char *phase = read_file_text(path, NULL); free(path);
            h.phase = phase ? atoi(phase) : H_PHASE_PRIMARY; free(phase);
        } else free(previous);
    }
    agent_statusf(a, "agent: hybrid supervisor; primary=%s expert=%s mode=%s; one resident model\n",
                  h.primary_model, h.dense_model, hybrid_mode_name(a->cfg.expert_mode));
    int rc = 0;
    struct sigaction prior = {0}, action = {0};
    if (a->cfg.prompt) {
        action.sa_handler = agent_sigint_handler; sigemptyset(&action.sa_mask);
        sigaction(SIGINT, &action, &prior);
        a->should_interrupt = hybrid_direct_interrupt;
    }
    if (h.job && h.phase != H_PHASE_DONE) rc = hybrid_recover(a);
    if (rc == 0) {
        if (a->cfg.prompt) rc = agent_hybrid_run(a, a->cfg.prompt);
        else rc = interactive_loop(a);
    }
    hybrid_close_process(a, true);
    if (a->cfg.prompt) sigaction(SIGINT, &prior, NULL);
    free(h.executable); free(h.primary_model); free(h.dense_model); free(h.root);
    free(h.primary_store); free(h.active); free(h.lock); free(h.job);
    a->hybrid_supervisor = NULL;
    return rc == 0 ? 0 : 1;
}

static int agent_hybrid_acquire_lock(agent_state *a) {
    if (!a->cfg.hybrid_lock) return -1;
    int fd = open(a->cfg.hybrid_lock, O_RDWR | O_CREAT, 0600);
    if (fd < 0) return -1;
    double end = agent_now_sec() + 10;
    while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno != EWOULDBLOCK || agent_now_sec() >= end) { close(fd); return -1; }
        struct pollfd p = {.fd = a->cfg.hybrid_fd, .events = POLLIN};
        if (poll(&p, 1, 100) > 0 && (p.revents & POLLHUP)) { close(fd); return -1; }
    }
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    return fd;
}

/* Model-free checks exercise the persisted state and actual IPC, not Metal mocks. */
static int agent_hybrid_selftest(void) {
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)) return 1;
    pid_t pid = fork();
    if (pid < 0) return 2;
    if (!pid) {
        close(sockets[0]);
        hybrid_header header = {HYBRID_MAGIC, H_OUTPUT, 4};
        const char *p = (const char *)&header;
        for (size_t i = 0; i < sizeof(header); i++) if (write(sockets[1], p+i, 1) != 1) _exit(1);
        for (int i = 0; i < 4; i++) if (write(sockets[1], &"ciao"[i], 1) != 1) _exit(1);
        close(sockets[1]); _exit(0);
    }
    close(sockets[1]); uint32_t type; size_t bytes;
    char *text = hybrid_receive(sockets[0], &type, &bytes);
    bool ok = text && type == H_OUTPUT && bytes == 4 && !strcmp(text, "ciao"); free(text);
    if (hybrid_receive(sockets[0], &type, &bytes)) ok = false;
    close(sockets[0]); int status = 0; waitpid(pid, &status, 0);
    if (!ok || status) return 3;
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)) return 4;
    hybrid_header header = {HYBRID_MAGIC, H_OUTPUT, HYBRID_MAX_FRAME+1};
    if (!hybrid_io(sockets[1], &header, sizeof(header), true)) return 5;
    if (hybrid_receive(sockets[0], &type, &bytes) || hybrid_send(sockets[1], H_OUTPUT, "", HYBRID_MAX_FRAME+1)) return 6;
    close(sockets[0]); close(sockets[1]);

    char directory[] = "/tmp/qw3-hybrid-selftest-XXXXXX";
    if (!mkdtemp(directory)) return 7;
    agent_state a = {0}; a.cfg.store_dir = directory; a.cfg.conversation = "test";
    a.cfg.hybrid_fd = 3; a.cfg.hybrid_job = directory;
    a.cfg.expert_model = "dense"; a.cfg.expert_mode = AGENT_EXPERT_AUTO;
    a.cfg.tools_enabled = true; a.cfg.expert_rounds = 2;
    a.transcript.len = 123; a.turn.round = 7; a.turn.guard_retry_credits = 1;
    a.turn.delegated = true; a.turn.tool_history.n_items = 1; a.turn.tool_history.hashes[0] = 789;
    a.cfg.sample.rng = 0xdeadbeef; a.turn_request = agent_strdup("vincoli integri: è α");
    a.last_read_path = agent_strdup("fixture.py"); a.last_read_next = 19;
    a.source_reads[0].path = agent_strdup("source.py"); a.source_reads[0].chunks = 2; a.source_reads[0].lines = 50;
    a.edit_verification_pending = true; a.hybrid_generated = 88;
    if (!hybrid_save_runtime(&a)) return 8;
    agent_state restored = {0}; restored.cfg = a.cfg; restored.transcript.len = 123;
    if (!hybrid_load_runtime(&restored) || restored.cfg.sample.rng != 0xdeadbeef ||
        restored.turn.round != 7 || !restored.turn.delegated || !restored.edit_verification_pending ||
        restored.hybrid_generated != 88 || strcmp(restored.turn_request, a.turn_request) ||
        restored.turn.tool_history.hashes[0] != 789 || restored.source_reads[0].lines != 50) return 9;
    restored.transcript.len = 122;
    if (hybrid_load_runtime(&restored)) return 10;
    char *path = hybrid_runtime_path(&a);
    if (!hybrid_atomic(path, "bad", 3) || hybrid_load_runtime(&restored)) return 11;
    unlink(path); free(path);
    free(restored.turn_request); free(restored.last_read_path); agent_reset_source_read_budget(&restored);
    free(a.turn_request); a.turn_request = NULL; free(a.last_read_path); agent_reset_source_read_budget(&a);

    tool_call_list calls = {.n_calls = 1}; tool_call *call = &calls.calls[0];
    snprintf(call->name, sizeof(call->name), "delegate_dense");
    const char *names[] = {"task", "reason", "context", "criteria"};
    call->n_params = 4;
    for (int i=0;i<4;i++) { snprintf(call->params[i].name, sizeof(call->params[i].name), "%s", names[i]); call->params[i].value = "evidence"; }
    a.turn.delegated = false;
    if (hybrid_delegation_error(&a, &calls, call)) return 12;
    a.turn_request = agent_strdup("analizza il progetto nella cartella corrente");
    if (!hybrid_delegation_error(&a, &calls, call)) return 22;
    a.hybrid_forced = true;
    if (hybrid_delegation_error(&a, &calls, call)) return 23;
    a.hybrid_forced = false; free(a.turn_request); a.turn_request = NULL;
    a.cfg.hybrid_dense = true;
    if (!hybrid_delegation_error(&a, &calls, call)) return 13;
    a.cfg.hybrid_dense = false; a.cfg.expert_mode = AGENT_EXPERT_OFF;
    if (!hybrid_delegation_error(&a, &calls, call)) return 14;
    a.cfg.expert_mode = AGENT_EXPERT_MANUAL;
    if (!hybrid_delegation_error(&a, &calls, call)) return 15;
    a.cfg.expert_mode = AGENT_EXPERT_AUTO; a.turn.delegated = true;
    if (!hybrid_delegation_error(&a, &calls, call)) return 16;
    a.turn.delegated = false; calls.n_calls = 2;
    if (!hybrid_delegation_error(&a, &calls, call)) return 17;
    calls.n_calls = 1; call->n_params = 3;
    if (!hybrid_delegation_error(&a, &calls, call)) return 18;
    a.cfg.hybrid_dense = true;
    if (!agent_hybrid_tool_budget(&a) || !agent_hybrid_tool_budget(&a) || agent_hybrid_tool_budget(&a)) return 19;
    agent_message message = {.kind = AGENT_MESSAGE_TOOL, .content = "<tool_response>\nHybrid result ID: job-1\nOK"};
    a.messages.items = &message; a.messages.len = 1;
    if (!hybrid_result_applied(&a, "Hybrid result ID: job-1\nOK") || hybrid_result_applied(&a, "Hybrid result ID: job-10\nOK")) return 20;
    message.kind = AGENT_MESSAGE_USER;
    if (hybrid_result_applied(&a, "Hybrid result ID: job-1\nOK")) return 21;
    path = path_join(directory, "result-applied");
    const char *id = "Hybrid result ID: job-1\n";
    if (!hybrid_atomic(path, id, strlen(id)) || !hybrid_result_applied(&a, "Hybrid result ID: job-1\nOK")) return 27;
    unlink(path); free(path);
    tool_call shell = {.n_params = 1}; snprintf(shell.name, sizeof(shell.name), "bash");
    snprintf(shell.params[0].name, sizeof(shell.params[0].name), "cmd");
    shell.params[0].value = "python3 -c \"write_once()\" & sleep 1";
    path = store_path_ext(&a, a.cfg.conversation, ".hybrid-replay-blocks");
    if (!hybrid_atomic(path, "python3-c\"write_once()\"\n", strlen("python3-c\"write_once()\"\n")) ||
        !hybrid_replay_blocked(&a, &shell)) return 24;
    shell.params[0].value = "python3 -c \"read_only_check()\"";
    if (hybrid_replay_blocked(&a, &shell)) return 25;
    unlink(path); free(path); rmdir(directory);
    a.cfg.read_only = true;
    const char *denied[] = {"write", "edit", "bash", "semantic_search", "search_web", "visit_url"};
    for (size_t i=0;i<sizeof(denied)/sizeof(denied[0]);i++) {
        tool_call blocked = {0}; snprintf(blocked.name, sizeof(blocked.name), "%s", denied[i]);
        char *out = execute_one_tool(&a, &blocked);
        bool denied_ok = out && strstr(out, "disabled in read-only mode"); free(out);
        if (!denied_ok) return 26;
    }
    return 0;
}
