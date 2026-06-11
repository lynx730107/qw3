#ifndef CODE_NAVIGATION_H
#define CODE_NAVIGATION_H

typedef struct {
    char *json;
    int error_code;
    char error_message[512];
} ToolResult;

ToolResult get_skeleton(const char *file_path);
ToolResult get_skeleton_compact(const char *file_path);
ToolResult get_function(const char *file_path, const char *function_name);

void tool_result_free(ToolResult *result);

#endif