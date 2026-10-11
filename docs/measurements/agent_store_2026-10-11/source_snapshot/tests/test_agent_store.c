/* Exercise real retention code in a temporary store, without loading a model. */
#define main qw3_agent_client_main
#include "../qw3_agent.c"
#undef main
#include <utime.h>

static int fixture_write(const char *path, const char *text) {
    return write_file_bytes(path, text, strlen(text));
}

static int fixture_mtime(const char *path, time_t when) {
    struct utimbuf times = { .actime = when, .modtime = when };
    return utime(path, &times);
}

static int fixture_count(const char *dir, const char *prefix) {
    DIR *dp = opendir(dir);
    if (!dp) return -1;
    int count = 0;
    struct dirent *de;
    while ((de = readdir(dp))) {
        if (strncmp(de->d_name, prefix, strlen(prefix))) continue;
        char *path = path_join(dir, de->d_name);
        struct stat st;
        if (path && lstat(path, &st) == 0 && S_ISREG(st.st_mode)) count++;
        free(path);
    }
    closedir(dp);
    return count;
}

static int check_prefix_retention(const char *root) {
    char path[PATH_MAX], protected_path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/unrelated", root);
    if (fixture_write(path, "preserve me") != 0) return 1;
    snprintf(path, sizeof(path), "%s/.prefix-v1-directory", root);
    if (mkdir(path, 0700) != 0) return 2;
    for (int i = 0; i < 12; i++) {
        snprintf(path, sizeof(path), "%s/.prefix-v%d-%016llx.kvc", root,
                 i % 2, (unsigned long long)i);
        if (fixture_write(path, "cache") != 0 ||
            fixture_mtime(path, i == 11 ? 900 : 1000 + i) != 0) return 3;
        agent_prune_spills(root, ".prefix-v", path,
                           QW3_AGENT_PREFIX_CACHE_KEEP);
        if (fixture_count(root, ".prefix-v") != (i < 4 ? i + 1 : 4)) return 4;
        if (i == 11) snprintf(protected_path, sizeof(protected_path), "%s", path);
    }
    if (access(protected_path, F_OK) != 0) return 5;
    for (int i = 0; i < 12; i++) {
        snprintf(path, sizeof(path), "%s/.prefix-v%d-%016llx.kvc", root,
                 i % 2, (unsigned long long)i);
        if ((access(path, F_OK) == 0) != (i >= 8)) return 6;
    }
    snprintf(path, sizeof(path), "%s/unrelated", root);
    if (access(path, F_OK) != 0) return 7;
    snprintf(path, sizeof(path), "%s/.prefix-v1-directory", root);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) return 8;
    puts("PASS prefix retention: 4 files across versions; current cache protected; unrelated entries preserved");
    return 0;
}

static int check_context_retention(const char *root) {
    agent_state state = {0};
    state.cfg.store_dir = (char *)root;
    state.transcript.len = 5;
    char *paths[12] = {0};
    int rc = 0;
    if (!agent_message_ledger_add(&state.messages, AGENT_MESSAGE_TOOL,
                                 "tool", "retention fixture", 0, 5)) rc = 1;
    for (int i = 0; rc == 0 && i < 12; i++) {
        paths[i] = agent_archive_context(&state);
        if (!paths[i] || fixture_mtime(paths[i], 1000 + i) != 0) rc = 2;
    }
    char *dir = path_join(root, "spills");
    if (rc == 0 && fixture_count(dir, "context-") != 8) rc = 3;
    for (int i = 0; rc == 0 && i < 12; i++) {
        if ((access(paths[i], F_OK) == 0) != (i >= 4)) rc = 4;
    }
    if (rc == 0) {
        struct stat st;
        char *text = read_file_text(paths[11], NULL);
        if (stat(paths[11], &st) != 0 || (st.st_mode & 0777) != 0600 ||
            !text || !strstr(text, "retention fixture")) rc = 5;
        free(text);
    }
    for (int i = 0; i < 12; i++) free(paths[i]);
    free(dir);
    agent_message_ledger_clear(&state.messages);
    if (rc == 0) puts("PASS context retention: 12 archives rotate to the latest 8 with intact content and private mode");
    return rc;
}

static int check_history_file(const char *path, int first) {
    FILE *fp = fopen(path, "r");
    if (!fp) return 1;
    char line[64], expected[64];
    int count = 0, rc = 0;
    while (fgets(line, sizeof(line), fp)) {
        snprintf(expected, sizeof(expected), "prompt-%d\n", first + count);
        if (strcmp(line, expected)) rc = 2;
        count++;
    }
    if (count != 1000 || ferror(fp)) rc = 3;
    fclose(fp);
    return rc;
}

static int check_history_retention(const char *root) {
    char path[PATH_MAX], line[64];
    snprintf(path, sizeof(path), "%s/history", root);
    if (!linenoiseHistorySetMaxLen(1000)) return 1;
    for (int i = 0; i < 1012; i++) {
        snprintf(line, sizeof(line), "prompt-%d", i);
        if (!linenoiseHistoryAdd(line)) return 2;
    }
    if (linenoiseHistorySave(path) != 0 || check_history_file(path, 12)) return 3;
    if (!linenoiseHistoryAdd("prompt-1012")) return 4;
    (void)linenoiseHistoryAdd("prompt-1012");
    if (linenoiseHistorySave(path) != 0 || check_history_file(path, 13)) return 5;
    if (linenoiseHistoryLoad(path) != 0 || linenoiseHistorySave(path) != 0 ||
        check_history_file(path, 13)) return 6;
    puts("PASS history retention: latest 1000 entries; save overwrites; adjacent duplicates suppressed; reload bounded");
    return 0;
}

static void fixture_cleanup(const char *root) {
    DIR *dp = opendir(root);
    if (!dp) return;
    struct dirent *de;
    while ((de = readdir(dp))) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char *path = path_join(root, de->d_name);
        struct stat st;
        if (path && lstat(path, &st) == 0) {
            if (S_ISDIR(st.st_mode)) fixture_cleanup(path);
            else (void)unlink(path);
        }
        free(path);
    }
    closedir(dp);
    (void)rmdir(root);
}

int main(void) {
    char root[] = "/tmp/qw3-agent-store-test-XXXXXX";
    if (!mkdtemp(root)) return 1;
    int rc = check_prefix_retention(root);
    if (!rc) rc = check_context_retention(root);
    if (!rc) rc = check_history_retention(root);
    fixture_cleanup(root);
    if (rc) fprintf(stderr, "test-agent-store: FAIL case %d\n", rc);
    else puts("test-agent-store: ok (no model; temporary store only)");
    return rc;
}
