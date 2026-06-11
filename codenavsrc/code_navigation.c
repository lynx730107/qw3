#include "code_navigation.h"

#include <tree_sitter/api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

/*
    Questo simbolo viene fornito da:

        tree-sitter-c/src/parser.c

    Quando compili/linki parser.c, questa funzione diventa disponibile.
*/
extern const TSLanguage *tree_sitter_c(void);
extern const TSLanguage *tree_sitter_python(void);
extern const TSLanguage *tree_sitter_cpp(void);
extern const TSLanguage *tree_sitter_javascript(void);
extern const TSLanguage *tree_sitter_go(void);
extern const TSLanguage *tree_sitter_c_sharp(void);
extern const TSLanguage *tree_sitter_ruby(void);
extern const TSLanguage *tree_sitter_objc(void);

/* ------------------------------------------------------------
   Query Tree-sitter embedded
   ------------------------------------------------------------ */

static const char *C_FUNCTIONS_QUERY =
    "(function_definition "
    "  declarator: (function_declarator "
    "    declarator: (identifier) @function.name)) @function.definition\n"

    "(function_definition "
    "  declarator: (pointer_declarator "
    "    declarator: (function_declarator "
    "      declarator: (identifier) @function.name))) @function.definition\n";

static const char *C_CALLS_QUERY =
    "(call_expression "
    "  function: (identifier) @call.name) @call.expression\n";

static const char *C_EXTRA_SKELETON_QUERY =
    "(preproc_include) @include\n"
    "(preproc_def) @macro\n"
    "(type_definition) @typedef\n"
    "(declaration) @global.declaration\n";

static const char *PY_FUNCTIONS_QUERY =
    "(function_definition name: (identifier) @function.name) @function.definition\n";

static const char *PY_CALLS_QUERY =
    "(call function: (identifier) @call.name) @call.expression\n"
    "(call function: (attribute attribute: (identifier) @call.name) @call.expression)\n";

static const char *PY_EXTRA_SKELETON_QUERY =
    "(import_statement) @include\n"
    "(import_from_statement) @include\n"
    "(future_import_statement) @include\n"
    "(assignment) @global.assignment\n"
    "(augmented_assignment) @global.assignment\n"
    "(type_alias_statement) @typedef\n";

static const char *CPP_FUNCTIONS_QUERY =
    "(function_definition "
    "  declarator: (function_declarator "
    "    declarator: (identifier) @function.name)) @function.definition\n"
    "(function_definition "
    "  declarator: (function_declarator "
    "    declarator: (qualified_identifier name: (identifier) @function.name))) @function.definition\n"
    "(function_definition "
    "  declarator: (pointer_declarator "
    "    declarator: (function_declarator "
    "      declarator: (identifier) @function.name))) @function.definition\n";

static const char *CPP_CALLS_QUERY =
    "(call_expression function: (identifier) @call.name) @call.expression\n"
    "(call_expression function: (field_expression field: (field_identifier) @call.name) @call.expression)\n";

static const char *CPP_EXTRA_SKELETON_QUERY =
    "(preproc_include) @include\n"
    "(preproc_def) @macro\n"
    "(type_definition) @typedef\n"
    "(declaration) @global.declaration\n";

static const char *JS_FUNCTIONS_QUERY =
    "(function_declaration name: (identifier) @function.name) @function.definition\n"
    "(generator_function_declaration name: (identifier) @function.name) @function.definition\n";

static const char *JS_CALLS_QUERY =
    "(call_expression function: (identifier) @call.name) @call.expression\n"
    "(call_expression function: (member_expression property: (property_identifier) @call.name) @call.expression)\n";

static const char *JS_EXTRA_SKELETON_QUERY =
    "(import_statement) @include\n"
    "(variable_declaration) @global.declaration\n"
    "(lexical_declaration) @global.declaration\n"
    "(class_declaration) @typedef\n";

static const char *GO_FUNCTIONS_QUERY =
    "(function_declaration name: (identifier) @function.name) @function.definition\n"
    "(method_declaration name: (field_identifier) @function.name) @function.definition\n";

static const char *GO_CALLS_QUERY =
    "(call_expression function: (identifier) @call.name) @call.expression\n"
    "(call_expression function: (selector_expression field: (field_identifier) @call.name) @call.expression)\n";

static const char *GO_EXTRA_SKELETON_QUERY =
    "(import_declaration) @include\n"
    "(type_declaration) @typedef\n"
    "(var_declaration) @global.declaration\n"
    "(const_declaration) @global.declaration\n";

static const char *CSHARP_FUNCTIONS_QUERY =
    "(method_declaration (identifier) @function.name) @function.definition\n";

static const char *CSHARP_CALLS_QUERY =
    "(invocation_expression (identifier) @call.name) @call.expression\n"
    "(invocation_expression (member_access_expression name: (identifier) @call.name) @call.expression)\n";

static const char *CSHARP_EXTRA_SKELETON_QUERY =
    "(using_directive) @include\n"
    "(field_declaration) @global.declaration\n";

static const char *RUBY_FUNCTIONS_QUERY =
    "(method name: (_) @function.name) @function.definition\n"
    "(singleton_method name: (_) @function.name) @function.definition\n";

static const char *RUBY_CALLS_QUERY =
    "(call method: (_) @call.name) @call.expression\n";

static const char *RUBY_EXTRA_SKELETON_QUERY =
    "(assignment) @global.declaration\n"
    "(class) @typedef\n"
    "(module) @typedef\n";

static const char *OBJC_FUNCTIONS_QUERY =
    "(function_definition "
    "  declarator: (function_declarator "
    "    declarator: (identifier) @function.name)) @function.definition\n"
    "(function_definition "
    "  declarator: (pointer_declarator "
    "    declarator: (function_declarator "
    "      declarator: (identifier) @function.name))) @function.definition\n"
    "(method_definition (identifier) @function.name) @function.definition\n"
    "(method_definition (keyword_declarator (identifier) @function.name)) @function.definition\n";

static const char *OBJC_CALLS_QUERY =
    "(call_expression function: (identifier) @call.name) @call.expression\n"
    "(call_expression function: (field_expression field: (field_identifier) @call.name) @call.expression)\n"
    "(message_expression method: (identifier) @call.name) @call.expression\n"
    "(selector_expression (identifier) @call.name) @call.expression\n";

static const char *OBJC_EXTRA_SKELETON_QUERY =
    "(preproc_include) @include\n"
    "(module_import) @include\n"
    "(preproc_def) @macro\n"
    "(type_definition) @typedef\n"
    "(declaration) @global.declaration\n"
    "(property_implementation) @global.declaration\n"
    "(compatibility_alias_declaration) @global.declaration\n";

/* ------------------------------------------------------------
   Strutture interne
   ------------------------------------------------------------ */

#define MAX_FUNCTIONS 512
#define MAX_CALLS_PER_FUNCTION 512
#define MAX_INCLUDES 256
#define MAX_MACROS 256
#define MAX_TYPEDEFS 512
#define MAX_GLOBAL_VARIABLES 512

typedef struct {
    TSNode node;
    char capture_name[128];
} CaptureNode;

typedef struct {
    TSPoint definition_start_point;
    TSPoint definition_end_point;
    char *name;
    char *text;
    char *signature;

    char *calls[MAX_CALLS_PER_FUNCTION];
    int call_count;
} FunctionInfo;

typedef struct {
    TSPoint start_point;
    TSPoint end_point;
    char *text;
} SkeletonItem;

typedef struct {
    const char *name;
    const char *extension;
    const TSLanguage *(*language_fn)(void);
    const char *functions_query;
    const char *calls_query;
    const char *extra_skeleton_query;
    int is_c_style_signature;
    int skip_class_scope_globals;
} LanguageDefinition;

static char *make_line_signature(const char *function_text);
static char *compact_include_text(const char *include_text);
static char *compact_macro_name(const char *macro_text);
static char *compact_typedef_label(const char *typedef_text);
static const char *basename_from_path(const char *path);
static char *compact_signature(const char *signature);
static const LanguageDefinition *detect_language_from_path(const char *file_path);
static const LanguageDefinition *cpp_language_definition(void);
static const LanguageDefinition *javascript_language_definition(void);
static const LanguageDefinition *go_language_definition(void);
static const LanguageDefinition *csharp_language_definition(void);
static const LanguageDefinition *ruby_language_definition(void);
static const LanguageDefinition *objc_language_definition(void);

static const LanguageDefinition *c_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "c";
        lang.extension = ".c";
        lang.language_fn = tree_sitter_c;
        lang.functions_query = C_FUNCTIONS_QUERY;
        lang.calls_query = C_CALLS_QUERY;
        lang.extra_skeleton_query = C_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 1;
        lang.skip_class_scope_globals = 0;
        initialized = 1;
    }

    return &lang;
}

static const LanguageDefinition *python_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "python";
        lang.extension = ".py";
        lang.language_fn = tree_sitter_python;
        lang.functions_query = PY_FUNCTIONS_QUERY;
        lang.calls_query = PY_CALLS_QUERY;
        lang.extra_skeleton_query = PY_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 0;
        lang.skip_class_scope_globals = 1;
        initialized = 1;
    }

    return &lang;
}

static const LanguageDefinition *cpp_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "cpp";
        lang.extension = ".cpp";
        lang.language_fn = tree_sitter_cpp;
        lang.functions_query = CPP_FUNCTIONS_QUERY;
        lang.calls_query = CPP_CALLS_QUERY;
        lang.extra_skeleton_query = CPP_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 1;
        lang.skip_class_scope_globals = 0;
        initialized = 1;
    }

    return &lang;
}

static const LanguageDefinition *javascript_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "javascript";
        lang.extension = ".js";
        lang.language_fn = tree_sitter_javascript;
        lang.functions_query = JS_FUNCTIONS_QUERY;
        lang.calls_query = JS_CALLS_QUERY;
        lang.extra_skeleton_query = JS_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 0;
        lang.skip_class_scope_globals = 1;
        initialized = 1;
    }

    return &lang;
}

static const LanguageDefinition *go_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "go";
        lang.extension = ".go";
        lang.language_fn = tree_sitter_go;
        lang.functions_query = GO_FUNCTIONS_QUERY;
        lang.calls_query = GO_CALLS_QUERY;
        lang.extra_skeleton_query = GO_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 0;
        lang.skip_class_scope_globals = 0;
        initialized = 1;
    }

    return &lang;
}

static const LanguageDefinition *csharp_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "csharp";
        lang.extension = ".cs";
        lang.language_fn = tree_sitter_c_sharp;
        lang.functions_query = CSHARP_FUNCTIONS_QUERY;
        lang.calls_query = CSHARP_CALLS_QUERY;
        lang.extra_skeleton_query = CSHARP_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 0;
        lang.skip_class_scope_globals = 0;
        initialized = 1;
    }

    return &lang;
}

static const LanguageDefinition *ruby_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "ruby";
        lang.extension = ".rb";
        lang.language_fn = tree_sitter_ruby;
        lang.functions_query = RUBY_FUNCTIONS_QUERY;
        lang.calls_query = RUBY_CALLS_QUERY;
        lang.extra_skeleton_query = RUBY_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 0;
        lang.skip_class_scope_globals = 1;
        initialized = 1;
    }

    return &lang;
}

static const LanguageDefinition *objc_language_definition(void) {
    static LanguageDefinition lang;
    static int initialized = 0;

    if (!initialized) {
        lang.name = "objc";
        lang.extension = ".m";
        lang.language_fn = tree_sitter_objc;
        lang.functions_query = OBJC_FUNCTIONS_QUERY;
        lang.calls_query = OBJC_CALLS_QUERY;
        lang.extra_skeleton_query = OBJC_EXTRA_SKELETON_QUERY;
        lang.is_c_style_signature = 1;
        lang.skip_class_scope_globals = 1;
        initialized = 1;
    }

    return &lang;
}

/* ------------------------------------------------------------
   Utility memoria risultato
   ------------------------------------------------------------ */

void tool_result_free(ToolResult *result) {
    if (!result) {
        return;
    }

    free(result->json);
    result->json = NULL;
    result->error_code = 0;
    result->error_message[0] = '\0';
}

static ToolResult make_error(int code, const char *message) {
    ToolResult r;
    r.json = NULL;
    r.error_code = code;

    snprintf(
        r.error_message,
        sizeof(r.error_message),
        "%s",
        message ? message : "Unknown error"
    );

    return r;
}

/* ------------------------------------------------------------
   Lettura file
   ------------------------------------------------------------ */

static char *read_file(const char *path, uint32_t *size_out) {
    FILE *f = fopen(path, "rb");

    if (!f) {
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }

    long size = ftell(f);

    if (size < 0) {
        fclose(f);
        return NULL;
    }

    rewind(f);

    char *buffer = (char *)malloc((size_t)size + 1);

    if (!buffer) {
        fclose(f);
        return NULL;
    }

    size_t read_size = fread(buffer, 1, (size_t)size, f);
    fclose(f);

    buffer[read_size] = '\0';

    if (size_out) {
        *size_out = (uint32_t)read_size;
    }

    return buffer;
}

static char *substr_dup(
    const char *source,
    uint32_t source_size,
    uint32_t start_byte,
    uint32_t end_byte
) {
    if (!source || end_byte < start_byte) {
        return NULL;
    }

    if (end_byte > source_size || start_byte > source_size) {
        return NULL;
    }

    uint32_t len = end_byte - start_byte;

    char *out = (char *)malloc((size_t)len + 1);

    if (!out) {
        return NULL;
    }

    memcpy(out, source + start_byte, len);
    out[len] = '\0';

    return out;
}

/* ------------------------------------------------------------
   JSON builder minimale
   ------------------------------------------------------------ */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} JsonBuffer;

static int jb_init(JsonBuffer *jb) {
    jb->cap = 4096;
    jb->len = 0;
    jb->data = (char *)malloc(jb->cap);

    if (!jb->data) {
        return 0;
    }

    jb->data[0] = '\0';
    return 1;
}

static int jb_reserve(JsonBuffer *jb, size_t extra) {
    if (jb->len + extra + 1 <= jb->cap) {
        return 1;
    }

    size_t new_cap = jb->cap;

    while (jb->len + extra + 1 > new_cap) {
        new_cap *= 2;
    }

    char *new_data = (char *)realloc(jb->data, new_cap);

    if (!new_data) {
        return 0;
    }

    jb->data = new_data;
    jb->cap = new_cap;

    return 1;
}

static int jb_append(JsonBuffer *jb, const char *s) {
    if (!s) {
        s = "";
    }

    size_t n = strlen(s);

    if (!jb_reserve(jb, n)) {
        return 0;
    }

    memcpy(jb->data + jb->len, s, n);
    jb->len += n;
    jb->data[jb->len] = '\0';

    return 1;
}

static int jb_append_char(JsonBuffer *jb, char c) {
    if (!jb_reserve(jb, 1)) {
        return 0;
    }

    jb->data[jb->len++] = c;
    jb->data[jb->len] = '\0';

    return 1;
}

static int jb_append_int(JsonBuffer *jb, uint32_t value) {
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "%u", value);
    return jb_append(jb, tmp);
}

static int jb_append_json_string(JsonBuffer *jb, const char *s) {
    if (!jb_append_char(jb, '"')) {
        return 0;
    }

    if (!s) {
        s = "";
    }

    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;

        switch (c) {
            case '\\':
                if (!jb_append(jb, "\\\\")) return 0;
                break;

            case '"':
                if (!jb_append(jb, "\\\"")) return 0;
                break;

            case '\n':
                if (!jb_append(jb, "\\n")) return 0;
                break;

            case '\r':
                if (!jb_append(jb, "\\r")) return 0;
                break;

            case '\t':
                if (!jb_append(jb, "\\t")) return 0;
                break;

            default:
                if (c < 32) {
                    char tmp[16];
                    snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                    if (!jb_append(jb, tmp)) return 0;
                } else {
                    if (!jb_append_char(jb, (char)c)) return 0;
                }
                break;
        }
    }

    return jb_append_char(jb, '"');
}

static int jb_append_range_points(JsonBuffer *jb, TSPoint start, TSPoint end) {

    if (!jb_append(jb, "{")) return 0;

    if (!jb_append(jb, "\"start_row\":")) return 0;
    if (!jb_append_int(jb, start.row)) return 0;

    if (!jb_append(jb, ",\"start_col\":")) return 0;
    if (!jb_append_int(jb, start.column)) return 0;

    if (!jb_append(jb, ",\"end_row\":")) return 0;
    if (!jb_append_int(jb, end.row)) return 0;

    if (!jb_append(jb, ",\"end_col\":")) return 0;
    if (!jb_append_int(jb, end.column)) return 0;

    if (!jb_append(jb, "}")) return 0;

    return 1;
}

/* ------------------------------------------------------------
   Tree-sitter helpers
   ------------------------------------------------------------ */

static int node_contains(TSNode outer, TSNode inner) {
    uint32_t os = ts_node_start_byte(outer);
    uint32_t oe = ts_node_end_byte(outer);
    uint32_t is = ts_node_start_byte(inner);
    uint32_t ie = ts_node_end_byte(inner);

    return is >= os && ie <= oe;
}

static int collect_captures(
    const TSLanguage *language,
    TSNode root,
    const char *query_source,
    CaptureNode **captures_out,
    char *error_message,
    size_t error_message_size
) {
    uint32_t error_offset = 0;
    TSQueryError error_type = TSQueryErrorNone;

    TSQuery *query = ts_query_new(
        language,
        query_source,
        (uint32_t)strlen(query_source),
        &error_offset,
        &error_type
    );

    if (!query) {
        snprintf(
            error_message,
            error_message_size,
            "TSQuery error at offset %u, type %d",
            error_offset,
            (int)error_type
        );

        return -1;
    }

    TSQueryCursor *cursor = ts_query_cursor_new();

    if (!cursor) {
        ts_query_delete(query);
        snprintf(error_message, error_message_size, "Cannot create TSQueryCursor");
        return -1;
    }

    ts_query_cursor_exec(cursor, query, root);

    size_t capacity = 2048;
    size_t count = 0;
    CaptureNode *captures = (CaptureNode *)malloc(sizeof(CaptureNode) * capacity);

    if (!captures) {
        ts_query_cursor_delete(cursor);
        ts_query_delete(query);
        snprintf(error_message, error_message_size, "Cannot allocate captures");
        return -1;
    }

    TSQueryMatch match;

    while (ts_query_cursor_next_match(cursor, &match)) {
        for (uint32_t i = 0; i < match.capture_count; i++) {
            if (count >= capacity) {
                size_t new_capacity = capacity * 2;
                CaptureNode *new_captures = (CaptureNode *)realloc(captures, sizeof(CaptureNode) * new_capacity);

                if (!new_captures) {
                    free(captures);
                    ts_query_cursor_delete(cursor);
                    ts_query_delete(query);
                    snprintf(error_message, error_message_size, "Cannot grow captures");
                    return -1;
                }

                captures = new_captures;
                capacity = new_capacity;
            }

            TSQueryCapture cap = match.captures[i];

            uint32_t name_len = 0;
            const char *capture_name = ts_query_capture_name_for_id(
                query,
                cap.index,
                &name_len
            );

            captures[count].node = cap.node;

            uint32_t n = name_len;

            if (n >= sizeof(captures[count].capture_name)) {
                n = sizeof(captures[count].capture_name) - 1;
            }

            memcpy(captures[count].capture_name, capture_name, n);
            captures[count].capture_name[n] = '\0';

            count++;
        }
    }

    ts_query_cursor_delete(cursor);
    ts_query_delete(query);

    *captures_out = captures;

    if (count > (size_t)INT32_MAX) {
        free(captures);
        snprintf(error_message, error_message_size, "Too many captures");
        return -1;
    }

    return (int)count;
}

static char *make_signature(const char *function_text) {
    if (!function_text) {
        return NULL;
    }

    const char *brace = strchr(function_text, '{');

    size_t raw_len;

    if (brace) {
        raw_len = (size_t)(brace - function_text);
    } else {
        raw_len = strlen(function_text);
    }

    char *tmp = (char *)malloc(raw_len + 1);

    if (!tmp) {
        return NULL;
    }

    memcpy(tmp, function_text, raw_len);
    tmp[raw_len] = '\0';

    char *out = (char *)malloc(raw_len + 1);

    if (!out) {
        free(tmp);
        return NULL;
    }

    size_t j = 0;
    int in_space = 0;

    for (size_t i = 0; tmp[i]; i++) {
        unsigned char c = (unsigned char)tmp[i];

        if (isspace(c)) {
            if (!in_space) {
                out[j++] = ' ';
                in_space = 1;
            }
        } else {
            out[j++] = (char)c;
            in_space = 0;
        }
    }

    while (j > 0 && isspace((unsigned char)out[j - 1])) {
        j--;
    }

    out[j] = '\0';

    free(tmp);
    return out;
}

static int call_already_exists(FunctionInfo *fn, const char *call_name) {
    for (int i = 0; i < fn->call_count; i++) {
        if (strcmp(fn->calls[i], call_name) == 0) {
            return 1;
        }
    }

    return 0;
}

static void function_info_free(FunctionInfo *fn) {
    if (!fn) {
        return;
    }

    free(fn->name);
    free(fn->text);
    free(fn->signature);

    for (int i = 0; i < fn->call_count; i++) {
        free(fn->calls[i]);
    }

    memset(fn, 0, sizeof(*fn));
}

static void skeleton_item_free(SkeletonItem *item) {
    if (!item) {
        return;
    }

    free(item->text);
    item->text = NULL;
    memset(item, 0, sizeof(*item));
}

static int node_has_ancestor_type(TSNode node, const char *type_name) {
    if (!type_name) {
        return 0;
    }

    TSNode current = ts_node_parent(node);

    while (!ts_node_is_null(current)) {
        const char *current_type = ts_node_type(current);

        if (current_type && strcmp(current_type, type_name) == 0) {
            return 1;
        }

        current = ts_node_parent(current);
    }

    return 0;
}

static int is_probable_function_prototype(const char *declaration_text) {
    if (!declaration_text) {
        return 0;
    }

    const char *semi = strchr(declaration_text, ';');

    if (!semi) {
        return 0;
    }

    const char *open = strchr(declaration_text, '(');
    const char *close = strchr(declaration_text, ')');

    if (!open || !close || open > semi || close > semi) {
        return 0;
    }

    if (strchr(declaration_text, '=')) {
        return 0;
    }

    return 1;
}

/* ------------------------------------------------------------
   Parsing file e costruzione FunctionInfo[]
   ------------------------------------------------------------ */

static ToolResult analyze_source_file(
    const char *file_path,
    const LanguageDefinition *lang,
    FunctionInfo *functions,
    int *function_count_out,
    SkeletonItem *includes,
    int *include_count_out,
    SkeletonItem *macros,
    int *macro_count_out,
    SkeletonItem *global_variables,
    int *global_variable_count_out,
    SkeletonItem *typedefs,
    int *typedef_count_out
) {
    if (!lang || !lang->language_fn || !lang->functions_query || !lang->calls_query) {
        return make_error(11, "Unsupported language configuration");
    }

    *function_count_out = 0;

    if (include_count_out) {
        *include_count_out = 0;
    }

    if (macro_count_out) {
        *macro_count_out = 0;
    }

    if (global_variable_count_out) {
        *global_variable_count_out = 0;
    }

    if (typedef_count_out) {
        *typedef_count_out = 0;
    }

    uint32_t source_size = 0;
    char *source = read_file(file_path, &source_size);

    if (!source) {
        return make_error(1, "Cannot read source file");
    }

    const TSLanguage *language = lang->language_fn();

    TSParser *parser = ts_parser_new();

    if (!parser) {
        free(source);
        return make_error(2, "Cannot create TSParser");
    }

    if (!ts_parser_set_language(parser, language)) {
        ts_parser_delete(parser);
        free(source);
        return make_error(3, "Cannot set parser language");
    }

    TSTree *tree = ts_parser_parse_string(
        parser,
        NULL,
        source,
        source_size
    );

    if (!tree) {
        ts_parser_delete(parser);
        free(source);
        return make_error(4, "Cannot parse source");
    }

    TSNode root = ts_tree_root_node(tree);

    char query_error[512] = {0};

    CaptureNode *function_captures = NULL;

    int function_capture_count = collect_captures(
        language,
        root,
        lang->functions_query,
        &function_captures,
        query_error,
        sizeof(query_error)
    );

    if (function_capture_count < 0) {
        free(function_captures);
        ts_tree_delete(tree);
        ts_parser_delete(parser);
        free(source);
        return make_error(5, query_error);
    }

    CaptureNode *call_captures = NULL;

    int call_capture_count = collect_captures(
        language,
        root,
        lang->calls_query,
        &call_captures,
        query_error,
        sizeof(query_error)
    );

    if (call_capture_count < 0) {
        free(call_captures);
        free(function_captures);
        ts_tree_delete(tree);
        ts_parser_delete(parser);
        free(source);
        return make_error(6, query_error);
    }

    int need_extras =
        includes && include_count_out &&
        macros && macro_count_out &&
        global_variables && global_variable_count_out &&
        typedefs && typedef_count_out;

    CaptureNode *extra_captures = NULL;
    int extra_capture_count = 0;

    if (need_extras) {
        extra_captures = NULL;

        if (!lang->extra_skeleton_query) {
            free(extra_captures);
            free(call_captures);
            free(function_captures);
            ts_tree_delete(tree);
            ts_parser_delete(parser);
            free(source);
            return make_error(12, "Missing extra skeleton query");
        }

        extra_capture_count = collect_captures(
            language,
            root,
            lang->extra_skeleton_query,
            &extra_captures,
            query_error,
            sizeof(query_error)
        );

        if (extra_capture_count < 0) {
            free(extra_captures);
            free(call_captures);
            free(function_captures);
            ts_tree_delete(tree);
            ts_parser_delete(parser);
            free(source);
            return make_error(10, query_error);
        }
    }

    int function_count = 0;

    for (int i = 0; i < function_capture_count; i++) {
        if (strcmp(function_captures[i].capture_name, "function.definition") != 0) {
            continue;
        }

        if (function_count >= MAX_FUNCTIONS) {
            break;
        }

        TSNode definition_node = function_captures[i].node;
        TSNode name_node = {0};
        int found_name = 0;

        for (int j = 0; j < function_capture_count; j++) {
            if (strcmp(function_captures[j].capture_name, "function.name") != 0) {
                continue;
            }

            if (node_contains(definition_node, function_captures[j].node)) {
                name_node = function_captures[j].node;
                found_name = 1;
                break;
            }
        }

        if (!found_name) {
            continue;
        }

        FunctionInfo *fn = &functions[function_count];
        memset(fn, 0, sizeof(*fn));

        fn->definition_start_point = ts_node_start_point(definition_node);
        fn->definition_end_point = ts_node_end_point(definition_node);

        fn->name = substr_dup(
            source,
            source_size,
            ts_node_start_byte(name_node),
            ts_node_end_byte(name_node)
        );

        fn->text = substr_dup(
            source,
            source_size,
            ts_node_start_byte(definition_node),
            ts_node_end_byte(definition_node)
        );

        fn->signature = lang->is_c_style_signature ? make_signature(fn->text) : make_line_signature(fn->text);

        if (!fn->name || !fn->text || !fn->signature) {
            function_info_free(fn);
            continue;
        }

        for (int k = 0; k < call_capture_count; k++) {
            if (strcmp(call_captures[k].capture_name, "call.name") != 0) {
                continue;
            }

            if (!node_contains(definition_node, call_captures[k].node)) {
                continue;
            }

            if (fn->call_count >= MAX_CALLS_PER_FUNCTION) {
                break;
            }

            char *call_name = substr_dup(
                source,
                source_size,
                ts_node_start_byte(call_captures[k].node),
                ts_node_end_byte(call_captures[k].node)
            );

            if (!call_name) {
                continue;
            }

            if (call_already_exists(fn, call_name)) {
                free(call_name);
                continue;
            }

            fn->calls[fn->call_count++] = call_name;
        }

        function_count++;
    }

    *function_count_out = function_count;

    if (need_extras) {
        int include_count = 0;
        int macro_count = 0;
        int global_variable_count = 0;
        int typedef_count = 0;

        for (int i = 0; i < extra_capture_count; i++) {
            TSNode node = extra_captures[i].node;

            if (strcmp(extra_captures[i].capture_name, "include") == 0) {
                if (include_count >= MAX_INCLUDES) {
                    continue;
                }

                char *text = substr_dup(
                    source,
                    source_size,
                    ts_node_start_byte(node),
                    ts_node_end_byte(node)
                );

                if (!text) {
                    continue;
                }

                includes[include_count].start_point = ts_node_start_point(node);
                includes[include_count].end_point = ts_node_end_point(node);
                includes[include_count].text = text;
                include_count++;
                continue;
            }

            if (strcmp(extra_captures[i].capture_name, "macro") == 0) {
                if (macro_count >= MAX_MACROS) {
                    continue;
                }

                if (node_has_ancestor_type(node, "function_definition")) {
                    continue;
                }

                char *text = substr_dup(
                    source,
                    source_size,
                    ts_node_start_byte(node),
                    ts_node_end_byte(node)
                );

                if (!text) {
                    continue;
                }

                macros[macro_count].start_point = ts_node_start_point(node);
                macros[macro_count].end_point = ts_node_end_point(node);
                macros[macro_count].text = text;
                macro_count++;
                continue;
            }

            if (strcmp(extra_captures[i].capture_name, "typedef") == 0) {
                if (typedef_count >= MAX_TYPEDEFS) {
                    continue;
                }

                if (node_has_ancestor_type(node, "function_definition")) {
                    continue;
                }

                char *text = substr_dup(
                    source,
                    source_size,
                    ts_node_start_byte(node),
                    ts_node_end_byte(node)
                );

                if (!text) {
                    continue;
                }

                typedefs[typedef_count].start_point = ts_node_start_point(node);
                typedefs[typedef_count].end_point = ts_node_end_point(node);
                typedefs[typedef_count].text = text;
                typedef_count++;
                continue;
            }

            if (
                strcmp(extra_captures[i].capture_name, "global.declaration") == 0 ||
                strcmp(extra_captures[i].capture_name, "global.assignment") == 0
            ) {
                if (global_variable_count >= MAX_GLOBAL_VARIABLES) {
                    continue;
                }

                if (node_has_ancestor_type(node, "function_definition")) {
                    continue;
                }

                if (lang->skip_class_scope_globals) {
                    if (node_has_ancestor_type(node, "class_definition")) {
                        continue;
                    }

                    if (node_has_ancestor_type(node, "class_declaration")) {
                        continue;
                    }

                    if (node_has_ancestor_type(node, "class")) {
                        continue;
                    }
                }

                char *text = substr_dup(
                    source,
                    source_size,
                    ts_node_start_byte(node),
                    ts_node_end_byte(node)
                );

                if (!text) {
                    continue;
                }

                if (lang->is_c_style_signature && (strncmp(text, "typedef", 7) == 0 || is_probable_function_prototype(text))) {
                    free(text);
                    continue;
                }

                global_variables[global_variable_count].start_point = ts_node_start_point(node);
                global_variables[global_variable_count].end_point = ts_node_end_point(node);
                global_variables[global_variable_count].text = text;
                global_variable_count++;
            }
        }

        *include_count_out = include_count;
        *macro_count_out = macro_count;
        *global_variable_count_out = global_variable_count;
        *typedef_count_out = typedef_count;
    }

    free(extra_captures);
    free(call_captures);
    free(function_captures);
    ts_tree_delete(tree);
    ts_parser_delete(parser);
    free(source);

    ToolResult ok;
    ok.json = NULL;
    ok.error_code = 0;
    ok.error_message[0] = '\0';

    return ok;
}

/* ------------------------------------------------------------
   Tool: get_skeleton
   ------------------------------------------------------------ */

ToolResult get_skeleton(const char *file_path) {
    const LanguageDefinition *lang = detect_language_from_path(file_path);

    if (!lang) {
        return make_error(11, "Unsupported file extension. Supported: .c .h .py .cpp .cc .cxx .hpp .hh .hxx .js .mjs .cjs .go .cs .csx .sharp .rb");
    }

    FunctionInfo *functions = (FunctionInfo *)calloc(MAX_FUNCTIONS, sizeof(FunctionInfo));
    SkeletonItem *includes = (SkeletonItem *)calloc(MAX_INCLUDES, sizeof(SkeletonItem));
    SkeletonItem *macros = (SkeletonItem *)calloc(MAX_MACROS, sizeof(SkeletonItem));
    SkeletonItem *global_variables = (SkeletonItem *)calloc(MAX_GLOBAL_VARIABLES, sizeof(SkeletonItem));
    SkeletonItem *typedefs = (SkeletonItem *)calloc(MAX_TYPEDEFS, sizeof(SkeletonItem));

    if (!functions || !includes || !macros || !global_variables || !typedefs) {
        free(functions);
        free(includes);
        free(macros);
        free(global_variables);
        free(typedefs);
        return make_error(102, "Cannot allocate function table");
    }

    int function_count = 0;
    int include_count = 0;
    int macro_count = 0;
    int global_variable_count = 0;
    int typedef_count = 0;

    ToolResult analysis = analyze_source_file(
        file_path,
        lang,
        functions,
        &function_count,
        includes,
        &include_count,
        macros,
        &macro_count,
        global_variables,
        &global_variable_count,
        typedefs,
        &typedef_count
    );

    if (analysis.error_code != 0) {
        free(typedefs);
        free(global_variables);
        free(macros);
        free(includes);
        free(functions);
        return analysis;
    }

    JsonBuffer jb;

    if (!jb_init(&jb)) {
        for (int i = 0; i < function_count; i++) {
            function_info_free(&functions[i]);
        }

        for (int i = 0; i < include_count; i++) {
            skeleton_item_free(&includes[i]);
        }

        for (int i = 0; i < macro_count; i++) {
            skeleton_item_free(&macros[i]);
        }

        for (int i = 0; i < global_variable_count; i++) {
            skeleton_item_free(&global_variables[i]);
        }

        for (int i = 0; i < typedef_count; i++) {
            skeleton_item_free(&typedefs[i]);
        }

        free(typedefs);
        free(global_variables);
        free(macros);
        free(includes);
        free(functions);

        return make_error(100, "Cannot allocate JSON buffer");
    }

    jb_append(&jb, "{\n");
    jb_append(&jb, "  \"file\": ");
    jb_append_json_string(&jb, file_path);
    jb_append(&jb, ",\n");

    jb_append(&jb, "  \"language\": ");
    jb_append_json_string(&jb, lang->name);
    jb_append(&jb, ",\n");

    jb_append(&jb, "  \"includes\": [\n");

    for (int i = 0; i < include_count; i++) {
        if (i > 0) {
            jb_append(&jb, ",\n");
        }

        jb_append(&jb, "    {\n");
        jb_append(&jb, "      \"text\": ");
        jb_append_json_string(&jb, includes[i].text);
        jb_append(&jb, ",\n");
        jb_append(&jb, "      \"range\": ");
        jb_append_range_points(&jb, includes[i].start_point, includes[i].end_point);
        jb_append(&jb, "\n");
        jb_append(&jb, "    }");
    }

    jb_append(&jb, "\n");
    jb_append(&jb, "  ],\n");

    jb_append(&jb, "  \"macros\": [\n");

    for (int i = 0; i < macro_count; i++) {
        if (i > 0) {
            jb_append(&jb, ",\n");
        }

        jb_append(&jb, "    {\n");
        jb_append(&jb, "      \"text\": ");
        jb_append_json_string(&jb, macros[i].text);
        jb_append(&jb, ",\n");
        jb_append(&jb, "      \"range\": ");
        jb_append_range_points(&jb, macros[i].start_point, macros[i].end_point);
        jb_append(&jb, "\n");
        jb_append(&jb, "    }");
    }

    jb_append(&jb, "\n");
    jb_append(&jb, "  ],\n");

    jb_append(&jb, "  \"global_variables\": [\n");

    for (int i = 0; i < global_variable_count; i++) {
        if (i > 0) {
            jb_append(&jb, ",\n");
        }

        jb_append(&jb, "    {\n");
        jb_append(&jb, "      \"text\": ");
        jb_append_json_string(&jb, global_variables[i].text);
        jb_append(&jb, ",\n");
        jb_append(&jb, "      \"range\": ");
        jb_append_range_points(&jb, global_variables[i].start_point, global_variables[i].end_point);
        jb_append(&jb, "\n");
        jb_append(&jb, "    }");
    }

    jb_append(&jb, "\n");
    jb_append(&jb, "  ],\n");

    jb_append(&jb, "  \"typedefs\": [\n");

    for (int i = 0; i < typedef_count; i++) {
        if (i > 0) {
            jb_append(&jb, ",\n");
        }

        jb_append(&jb, "    {\n");
        jb_append(&jb, "      \"text\": ");
        jb_append_json_string(&jb, typedefs[i].text);
        jb_append(&jb, ",\n");
        jb_append(&jb, "      \"range\": ");
        jb_append_range_points(&jb, typedefs[i].start_point, typedefs[i].end_point);
        jb_append(&jb, "\n");
        jb_append(&jb, "    }");
    }

    jb_append(&jb, "\n");
    jb_append(&jb, "  ],\n");

    jb_append(&jb, "  \"functions\": [\n");

    for (int i = 0; i < function_count; i++) {
        FunctionInfo *fn = &functions[i];

        if (i > 0) {
            jb_append(&jb, ",\n");
        }

        jb_append(&jb, "    {\n");

        jb_append(&jb, "      \"name\": ");
        jb_append_json_string(&jb, fn->name);
        jb_append(&jb, ",\n");

        jb_append(&jb, "      \"signature\": ");
        jb_append_json_string(&jb, fn->signature);
        jb_append(&jb, ",\n");

        jb_append(&jb, "      \"range\": ");
        jb_append_range_points(&jb, fn->definition_start_point, fn->definition_end_point);
        jb_append(&jb, ",\n");

        jb_append(&jb, "      \"calls\": [");

        for (int c = 0; c < fn->call_count; c++) {
            if (c > 0) {
                jb_append(&jb, ", ");
            }

            jb_append_json_string(&jb, fn->calls[c]);
        }

        jb_append(&jb, "]\n");

        jb_append(&jb, "    }");
    }

    jb_append(&jb, "\n");
    jb_append(&jb, "  ]\n");
    jb_append(&jb, "}\n");

    for (int i = 0; i < function_count; i++) {
        function_info_free(&functions[i]);
    }

    for (int i = 0; i < include_count; i++) {
        skeleton_item_free(&includes[i]);
    }

    for (int i = 0; i < macro_count; i++) {
        skeleton_item_free(&macros[i]);
    }

    for (int i = 0; i < global_variable_count; i++) {
        skeleton_item_free(&global_variables[i]);
    }

    for (int i = 0; i < typedef_count; i++) {
        skeleton_item_free(&typedefs[i]);
    }

    free(typedefs);
    free(global_variables);
    free(macros);
    free(includes);
    free(functions);

    ToolResult result;
    result.json = jb.data;
    result.error_code = 0;
    result.error_message[0] = '\0';

    return result;
}

ToolResult get_skeleton_compact(const char *file_path) {
    const LanguageDefinition *lang = detect_language_from_path(file_path);

    if (!lang) {
        return make_error(11, "Unsupported file extension. Supported: .c .h .py .cpp .cc .cxx .hpp .hh .hxx .js .mjs .cjs .go .cs .csx .sharp .rb");
    }

    FunctionInfo *functions = (FunctionInfo *)calloc(MAX_FUNCTIONS, sizeof(FunctionInfo));
    SkeletonItem *includes = (SkeletonItem *)calloc(MAX_INCLUDES, sizeof(SkeletonItem));
    SkeletonItem *macros = (SkeletonItem *)calloc(MAX_MACROS, sizeof(SkeletonItem));
    SkeletonItem *global_variables = (SkeletonItem *)calloc(MAX_GLOBAL_VARIABLES, sizeof(SkeletonItem));
    SkeletonItem *typedefs = (SkeletonItem *)calloc(MAX_TYPEDEFS, sizeof(SkeletonItem));

    if (!functions || !includes || !macros || !global_variables || !typedefs) {
        free(functions);
        free(includes);
        free(macros);
        free(global_variables);
        free(typedefs);
        return make_error(102, "Cannot allocate function table");
    }

    int function_count = 0;
    int include_count = 0;
    int macro_count = 0;
    int global_variable_count = 0;
    int typedef_count = 0;

    ToolResult analysis = analyze_source_file(
        file_path,
        lang,
        functions,
        &function_count,
        includes,
        &include_count,
        macros,
        &macro_count,
        global_variables,
        &global_variable_count,
        typedefs,
        &typedef_count
    );

    if (analysis.error_code != 0) {
        free(typedefs);
        free(global_variables);
        free(macros);
        free(includes);
        free(functions);
        return analysis;
    }

    JsonBuffer jb;

    if (!jb_init(&jb)) {
        for (int i = 0; i < function_count; i++) {
            function_info_free(&functions[i]);
        }

        for (int i = 0; i < include_count; i++) {
            skeleton_item_free(&includes[i]);
        }

        for (int i = 0; i < macro_count; i++) {
            skeleton_item_free(&macros[i]);
        }

        for (int i = 0; i < global_variable_count; i++) {
            skeleton_item_free(&global_variables[i]);
        }

        for (int i = 0; i < typedef_count; i++) {
            skeleton_item_free(&typedefs[i]);
        }

        free(typedefs);
        free(global_variables);
        free(macros);
        free(includes);
        free(functions);

        return make_error(100, "Cannot allocate output buffer");
    }

    jb_append(&jb, "Skeleton of ");
    jb_append(&jb, basename_from_path(file_path));
    jb_append(&jb, " (");
    jb_append(&jb, lang->name);
    jb_append(&jb, "):\n\n");

    jb_append(&jb, "### Includes\n");
    if (include_count == 0) {
        jb_append(&jb, "- (none)\n\n");
    } else {
        for (int i = 0; i < include_count; i++) {
            char *value = compact_include_text(includes[i].text);
            jb_append(&jb, "- ");
            jb_append(&jb, value ? value : includes[i].text);
            jb_append(&jb, "\n");
            free(value);
        }
        jb_append(&jb, "\n");
    }

    jb_append(&jb, "### Macros\n");
    if (macro_count == 0) {
        jb_append(&jb, "- (none)\n\n");
    } else {
        for (int i = 0; i < macro_count; i++) {
            char *value = compact_macro_name(macros[i].text);
            jb_append(&jb, "- ");
            jb_append(&jb, value ? value : macros[i].text);
            jb_append(&jb, "\n");
            free(value);
        }
        jb_append(&jb, "\n");
    }

    jb_append(&jb, "### Types\n");
    if (typedef_count == 0) {
        jb_append(&jb, "- (none)\n\n");
    } else {
        for (int i = 0; i < typedef_count; i++) {
            char *value = compact_typedef_label(typedefs[i].text);
            jb_append(&jb, "- ");
            jb_append(&jb, value ? value : typedefs[i].text);
            jb_append(&jb, "\n");
            free(value);
        }
        jb_append(&jb, "\n");
    }

    jb_append(&jb, "### Globals\n");
    if (global_variable_count == 0) {
        jb_append(&jb, "- (none)\n\n");
    } else {
        for (int i = 0; i < global_variable_count; i++) {
            jb_append(&jb, "- ");
            jb_append(&jb, global_variables[i].text);
            jb_append(&jb, "\n");
        }
        jb_append(&jb, "\n");
    }

    jb_append(&jb, "### Functions\n");
    if (function_count == 0) {
        jb_append(&jb, "- (none)\n");
    } else {
        for (int i = 0; i < function_count; i++) {
            FunctionInfo *fn = &functions[i];
            char *sig = compact_signature(fn->signature);

            jb_append(&jb, "- ");
            jb_append(&jb, sig ? sig : fn->signature);
            jb_append(&jb, " calls: ");

            if (fn->call_count == 0) {
                jb_append(&jb, "(none)");
            } else {
                for (int c = 0; c < fn->call_count; c++) {
                    if (c > 0) {
                        jb_append(&jb, ", ");
                    }
                    jb_append(&jb, fn->calls[c]);
                }
            }

            jb_append(&jb, " [");
            jb_append_int(&jb, fn->definition_start_point.row);
            jb_append(&jb, ":");
            jb_append_int(&jb, fn->definition_start_point.column);
            jb_append(&jb, " - ");
            jb_append_int(&jb, fn->definition_end_point.row);
            jb_append(&jb, ":");
            jb_append_int(&jb, fn->definition_end_point.column);
            jb_append(&jb, "]\n");

            free(sig);
        }
    }

    for (int i = 0; i < function_count; i++) {
        function_info_free(&functions[i]);
    }

    for (int i = 0; i < include_count; i++) {
        skeleton_item_free(&includes[i]);
    }

    for (int i = 0; i < macro_count; i++) {
        skeleton_item_free(&macros[i]);
    }

    for (int i = 0; i < global_variable_count; i++) {
        skeleton_item_free(&global_variables[i]);
    }

    for (int i = 0; i < typedef_count; i++) {
        skeleton_item_free(&typedefs[i]);
    }

    free(typedefs);
    free(global_variables);
    free(macros);
    free(includes);
    free(functions);

    ToolResult result;
    result.json = jb.data;
    result.error_code = 0;
    result.error_message[0] = '\0';

    return result;
}

/* ------------------------------------------------------------
   Tool: get_function
   ------------------------------------------------------------ */

ToolResult get_function(const char *file_path, const char *function_name) {
    const LanguageDefinition *lang = detect_language_from_path(file_path);

    if (!lang) {
        return make_error(11, "Unsupported file extension. Supported: .c .h .py .cpp .cc .cxx .hpp .hh .hxx .js .mjs .cjs .go .cs .csx .sharp .rb");
    }

    FunctionInfo *functions = (FunctionInfo *)calloc(MAX_FUNCTIONS, sizeof(FunctionInfo));

    if (!functions) {
        return make_error(103, "Cannot allocate function table");
    }

    int function_count = 0;

    ToolResult analysis = analyze_source_file(
        file_path,
        lang,
        functions,
        &function_count,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
    );

    if (analysis.error_code != 0) {
        free(functions);
        return analysis;
    }

    FunctionInfo *target = NULL;

    for (int i = 0; i < function_count; i++) {
        if (strcmp(functions[i].name, function_name) == 0) {
            target = &functions[i];
            break;
        }
    }

    if (!target) {
        for (int i = 0; i < function_count; i++) {
            function_info_free(&functions[i]);
        }

        free(functions);

        ToolResult err = make_error(200, "Function not found");
        snprintf(
            err.error_message,
            sizeof(err.error_message),
            "Function not found: %s",
            function_name
        );

        return err;
    }

    JsonBuffer jb;

    if (!jb_init(&jb)) {
        for (int i = 0; i < function_count; i++) {
            function_info_free(&functions[i]);
        }
        free(functions);

        return make_error(101, "Cannot allocate JSON buffer");
    }

    jb_append(&jb, "{\n");

    jb_append(&jb, "  \"file\": ");
    jb_append_json_string(&jb, file_path);
    jb_append(&jb, ",\n");

    jb_append(&jb, "  \"language\": ");
    jb_append_json_string(&jb, lang->name);
    jb_append(&jb, ",\n");

    jb_append(&jb, "  \"symbol\": ");
    jb_append_json_string(&jb, target->name);
    jb_append(&jb, ",\n");

    jb_append(&jb, "  \"range\": ");
    jb_append_range_points(&jb, target->definition_start_point, target->definition_end_point);
    jb_append(&jb, ",\n");

    jb_append(&jb, "  \"text\": ");
    jb_append_json_string(&jb, target->text);
    jb_append(&jb, "\n");

    jb_append(&jb, "}\n");

    for (int i = 0; i < function_count; i++) {
        function_info_free(&functions[i]);
    }
    free(functions);

    ToolResult result;
    result.json = jb.data;
    result.error_code = 0;
    result.error_message[0] = '\0';

    return result;
}

static char *make_line_signature(const char *function_text) {
    if (!function_text) {
        return NULL;
    }

    size_t len = 0;

    while (function_text[len] && function_text[len] != '\n' && function_text[len] != '\r') {
        len++;
    }

    while (len > 0 && isspace((unsigned char)function_text[len - 1])) {
        len--;
    }

    char *out = (char *)malloc(len + 1);

    if (!out) {
        return NULL;
    }

    memcpy(out, function_text, len);
    out[len] = '\0';

    return out;
}

static char *compact_include_text(const char *include_text) {
    if (!include_text) {
        return NULL;
    }

    const char *p = include_text;

    while (*p && *p != '<' && *p != '"') {
        p++;
    }

    if (*p == '<') {
        const char *end = strchr(p, '>');

        if (!end) {
            return NULL;
        }

        size_t len = (size_t)(end - p + 1);
        char *out = (char *)malloc(len + 1);

        if (!out) {
            return NULL;
        }

        memcpy(out, p, len);
        out[len] = '\0';
        return out;
    }

    if (*p == '"') {
        const char *end = strchr(p + 1, '"');

        if (!end) {
            return NULL;
        }

        size_t len = (size_t)(end - p + 1);
        char *out = (char *)malloc(len + 1);

        if (!out) {
            return NULL;
        }

        memcpy(out, p, len);
        out[len] = '\0';
        return out;
    }

    return NULL;
}

static char *compact_macro_name(const char *macro_text) {
    if (!macro_text) {
        return NULL;
    }

    const char *def = strstr(macro_text, "define");

    if (!def) {
        return NULL;
    }

    const char *p = def + 6;

    while (*p && isspace((unsigned char)*p)) {
        p++;
    }

    if (!*p) {
        return NULL;
    }

    const char *start = p;

    while (*p && (isalnum((unsigned char)*p) || *p == '_')) {
        p++;
    }

    if (p == start) {
        return NULL;
    }

    size_t len = (size_t)(p - start);
    char *out = (char *)malloc(len + 1);

    if (!out) {
        return NULL;
    }

    memcpy(out, start, len);
    out[len] = '\0';
    return out;
}

static char *compact_typedef_label(const char *typedef_text) {
    if (!typedef_text) {
        return NULL;
    }

    const char *semi = strrchr(typedef_text, ';');
    const char *end = semi ? semi : typedef_text + strlen(typedef_text);
    const char *p = end;

    while (p > typedef_text && isspace((unsigned char)p[-1])) {
        p--;
    }

    const char *name_end = p;

    while (p > typedef_text && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) {
        p--;
    }

    if (p == name_end) {
        return NULL;
    }

    size_t name_len = (size_t)(name_end - p);
    size_t prefix_len = strlen("type ");
    char *out = (char *)malloc(prefix_len + name_len + 1);

    if (!out) {
        return NULL;
    }

    memcpy(out, "type ", prefix_len);
    memcpy(out + prefix_len, p, name_len);
    out[prefix_len + name_len] = '\0';

    return out;
}

static const char *basename_from_path(const char *path) {
    if (!path) {
        return "";
    }

    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    const char *base = path;

    if (slash && slash + 1 > base) {
        base = slash + 1;
    }

    if (backslash && backslash + 1 > base) {
        base = backslash + 1;
    }

    return base;
}

static char *compact_signature(const char *signature) {
    if (!signature) {
        return NULL;
    }

    const char *p = signature;

    if (strncmp(p, "static ", 7) == 0) {
        p += 7;
    }

    size_t len = strlen(p);
    char *out = (char *)malloc(len + 1);

    if (!out) {
        return NULL;
    }

    memcpy(out, p, len + 1);
    return out;
}

static const LanguageDefinition *detect_language_from_path(const char *file_path) {
    if (!file_path) {
        return NULL;
    }

    size_t len = strlen(file_path);

    if (len >= 2 && strcmp(file_path + len - 2, ".c") == 0) {
        return c_language_definition();
    }

    if (len >= 2 && strcmp(file_path + len - 2, ".h") == 0) {
        return c_language_definition();
    }

    if (len >= 3 && strcmp(file_path + len - 3, ".py") == 0) {
        return python_language_definition();
    }

    if (len >= 4 && strcmp(file_path + len - 4, ".cpp") == 0) {
        return cpp_language_definition();
    }

    if (len >= 3 && strcmp(file_path + len - 3, ".cc") == 0) {
        return cpp_language_definition();
    }

    if (len >= 4 && strcmp(file_path + len - 4, ".cxx") == 0) {
        return cpp_language_definition();
    }

    if (len >= 4 && strcmp(file_path + len - 4, ".hpp") == 0) {
        return cpp_language_definition();
    }

    if (len >= 3 && strcmp(file_path + len - 3, ".hh") == 0) {
        return cpp_language_definition();
    }

    if (len >= 4 && strcmp(file_path + len - 4, ".hxx") == 0) {
        return cpp_language_definition();
    }

    if (len >= 3 && strcmp(file_path + len - 3, ".js") == 0) {
        return javascript_language_definition();
    }

    if (len >= 4 && strcmp(file_path + len - 4, ".mjs") == 0) {
        return javascript_language_definition();
    }

    if (len >= 4 && strcmp(file_path + len - 4, ".cjs") == 0) {
        return javascript_language_definition();
    }

    if (len >= 3 && strcmp(file_path + len - 3, ".go") == 0) {
        return go_language_definition();
    }

    if (len >= 3 && strcmp(file_path + len - 3, ".cs") == 0) {
        return csharp_language_definition();
    }

    if (len >= 4 && strcmp(file_path + len - 4, ".csx") == 0) {
        return csharp_language_definition();
    }

    if (len >= 6 && strcmp(file_path + len - 6, ".sharp") == 0) {
        return csharp_language_definition();
    }

    if (len >= 3 && strcmp(file_path + len - 3, ".rb") == 0) {
        return ruby_language_definition();
    }

    if (len >= 2 && strcmp(file_path + len - 2, ".m") == 0) {
        return objc_language_definition();
    }

    if (len >= 5 && strcmp(file_path + len - 5, ".objc") == 0) {
        return objc_language_definition();
    }

    return NULL;
}
