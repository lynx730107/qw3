#include <stdio.h>
#include <string.h>

#include <dirent.h>

#include "code_navigation.h"

static int write_text_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");

    if (!f) {
        return 0;
    }

    size_t len = strlen(text);
    size_t written = fwrite(text, 1, len, f);
    fclose(f);

    return written == len;
}

static void print_usage(const char *program_name) {
    fprintf(stderr, "Uso:\n");
    fprintf(stderr, "  %s get_skeleton nomefile.{c,h,py,cpp,cc,cxx,hpp,hh,hxx,js,mjs,cjs,go,cs,csx,sharp,rb,m,objc} [--json] [--save output.txt|output.json]\n", program_name);
    fprintf(stderr, "  %s get_function nome_funzione [nomefile.{c,h,py,cpp,cc,cxx,hpp,hh,hxx,js,mjs,cjs,go,cs,csx,sharp,rb,m,objc}] [--save output.json]\n", program_name);
}

static int has_supported_extension(const char *file_name) {
    size_t len = strlen(file_name);

    if (len >= 2 && strcmp(file_name + len - 2, ".c") == 0) {
        return 1;
    }

    if (len >= 3 && strcmp(file_name + len - 3, ".py") == 0) {
        return 1;
    }

    if (len >= 2 && strcmp(file_name + len - 2, ".h") == 0) {
        return 1;
    }

    if (len >= 4 && strcmp(file_name + len - 4, ".cpp") == 0) {
        return 1;
    }

    if (len >= 3 && strcmp(file_name + len - 3, ".cc") == 0) {
        return 1;
    }

    if (len >= 4 && strcmp(file_name + len - 4, ".cxx") == 0) {
        return 1;
    }

    if (len >= 4 && strcmp(file_name + len - 4, ".hpp") == 0) {
        return 1;
    }

    if (len >= 3 && strcmp(file_name + len - 3, ".hh") == 0) {
        return 1;
    }

    if (len >= 4 && strcmp(file_name + len - 4, ".hxx") == 0) {
        return 1;
    }

    if (len >= 3 && strcmp(file_name + len - 3, ".js") == 0) {
        return 1;
    }

    if (len >= 4 && strcmp(file_name + len - 4, ".mjs") == 0) {
        return 1;
    }

    if (len >= 4 && strcmp(file_name + len - 4, ".cjs") == 0) {
        return 1;
    }

    if (len >= 3 && strcmp(file_name + len - 3, ".go") == 0) {
        return 1;
    }

    if (len >= 3 && strcmp(file_name + len - 3, ".cs") == 0) {
        return 1;
    }

    if (len >= 4 && strcmp(file_name + len - 4, ".csx") == 0) {
        return 1;
    }

    if (len >= 6 && strcmp(file_name + len - 6, ".sharp") == 0) {
        return 1;
    }

    if (len >= 3 && strcmp(file_name + len - 3, ".rb") == 0) {
        return 1;
    }

    if (len >= 2 && strcmp(file_name + len - 2, ".m") == 0) {
        return 1;
    }

    if (len >= 5 && strcmp(file_name + len - 5, ".objc") == 0) {
        return 1;
    }

    if (len < 2) {
        return 0;
    }

    return 0;
}

static ToolResult get_function_auto(const char *function_name) {
    ToolResult best;
    best.json = NULL;
    best.error_code = 200;
    snprintf(best.error_message, sizeof(best.error_message), "Function not found: %s", function_name);

    DIR *dir = opendir(".");

    if (!dir) {
        ToolResult err;
        err.json = NULL;
        err.error_code = 1;
        snprintf(err.error_message, sizeof(err.error_message), "Cannot open current directory");
        return err;
    }

    struct dirent *entry;
    int found_count = 0;
    char first_file[512] = {0};
    char second_file[512] = {0};

    while ((entry = readdir(dir)) != NULL) {
        const char *file_name = entry->d_name;

        if (!has_supported_extension(file_name)) {
            continue;
        }

        ToolResult current = get_function(file_name, function_name);

        if (current.error_code == 0) {
            found_count++;

            if (found_count == 1) {
                snprintf(first_file, sizeof(first_file), "%s", file_name);
                best = current;
            } else if (found_count == 2) {
                snprintf(second_file, sizeof(second_file), "%s", file_name);
                tool_result_free(&best);
                tool_result_free(&current);

                ToolResult ambiguous;
                ambiguous.json = NULL;
                ambiguous.error_code = 201;
                snprintf(
                    ambiguous.error_message,
                    sizeof(ambiguous.error_message),
                    "Function '%s' trovata in piu file: %s, %s",
                    function_name,
                    first_file,
                    second_file
                );

                closedir(dir);
                return ambiguous;
            } else {
                tool_result_free(&current);
            }
        } else {
            tool_result_free(&current);
        }
    }

    closedir(dir);

    return best;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "get_skeleton") == 0) {
        const char *file_path = argv[2];
        const char *save_path = NULL;
        int output_json = 0;

        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--json") == 0) {
                output_json = 1;
                continue;
            }

            if (strcmp(argv[i], "--save") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Argomento mancante dopo --save\n");
                    return 1;
                }

                save_path = argv[++i];
                continue;
            }

            fprintf(stderr, "Argomento non riconosciuto: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }

        ToolResult skeleton = output_json ? get_skeleton(file_path) : get_skeleton_compact(file_path);

        if (skeleton.error_code != 0) {
            fprintf(stderr, "get_skeleton error: %s\n", skeleton.error_message);
            tool_result_free(&skeleton);
            return 2;
        }

        printf("%s\n", skeleton.json);

        if (save_path && !write_text_file(save_path, skeleton.json)) {
            fprintf(stderr, "Errore nel salvataggio output su file: %s\n", save_path);
            tool_result_free(&skeleton);
            return 4;
        }

        tool_result_free(&skeleton);
        return 0;
    }

    if (strcmp(argv[1], "get_function") == 0) {
        const char *function_name = argv[2];
        const char *file_path = NULL;
        const char *save_path = NULL;

        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--save") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Argomento mancante dopo --save\n");
                    return 1;
                }

                save_path = argv[++i];
                continue;
            }

            if (!file_path) {
                file_path = argv[i];
                continue;
            }

            fprintf(stderr, "Troppi argomenti o non riconosciuti: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }

        ToolResult fn;

        if (file_path) {
            fn = get_function(file_path, function_name);
        } else {
            fn = get_function_auto(function_name);
        }

        if (fn.error_code != 0) {
            fprintf(stderr, "get_function error: %s\n", fn.error_message);
            tool_result_free(&fn);
            return 3;
        }

        printf("%s\n", fn.json);

        if (save_path && !write_text_file(save_path, fn.json)) {
            fprintf(stderr, "Errore nel salvataggio JSON su file: %s\n", save_path);
            tool_result_free(&fn);
            return 4;
        }

        tool_result_free(&fn);
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}
