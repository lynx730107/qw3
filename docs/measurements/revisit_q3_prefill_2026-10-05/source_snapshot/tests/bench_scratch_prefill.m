#include "scratch_candidates.inc"
#define main scratch_bench_main
#include "../qw3_bench.c"
#undef main

int main(int argc, char **argv) {
    @autoreleasepool {
        int rc = scratch_bench_main(argc, argv);
        fprintf(stderr, "private dispatches pair=0 add=0 gdn=%llu scratch_projection=%llu scratch_output=%llu copies_skipped=%llu\n",
#ifdef QW3_PRIVATE_WITH_GDN
            (unsigned long long)probe_gdn_calls,
#else
            0ull,
#endif
            (unsigned long long)scratch_projection_calls,
            (unsigned long long)scratch_output_calls,
            (unsigned long long)scratch_copies_skipped);
        return rc;
    }
}
