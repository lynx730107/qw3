#define QW3_PRIVATE_COPY_CONTROL 1
#include "native_prefill_validation.inc"
#define main probe_native_main
#include "../qw3_bench.c"
#undef main

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *restore = getenv("QW3_PRIVATE_RESTORE_COPIES");
        if (restore && strcmp(restore, "1")) return 2;
        scratch_select(restore ? 3 : 0);
        int rc = probe_native_main(argc, argv);
        fprintf(stderr, "private dispatches pair=0 add=0 gdn=0 scratch_projection=%llu scratch_output=%llu copies_skipped=%llu restore=%d\n",
            (unsigned long long)scratch_projection_calls,
            (unsigned long long)scratch_output_calls,
            (unsigned long long)scratch_copies_skipped, probe_restore_copies);
        return rc;
    }
}
