#include "scratch_candidates.inc"
#define main scratch_bench_main
#include "../qw3_bench.c"
#undef main

int main(int argc, char **argv) {
    @autoreleasepool {
        int rc = scratch_bench_main(argc, argv);
        fprintf(stderr, "private dispatches pair=0 add=0 gdn=0 scratch_projection=%llu scratch_output=%llu copies_skipped=%llu\n",
            (unsigned long long)scratch_projection_calls,
            (unsigned long long)scratch_output_calls,
            (unsigned long long)scratch_copies_skipped);
        return rc;
    }
}
