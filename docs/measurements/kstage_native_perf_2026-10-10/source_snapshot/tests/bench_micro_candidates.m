#include "micro_candidates.inc"
#define main micro_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc, char **argv) {
    @autoreleasepool {
        const char *mode = getenv("QW3_PRIVATE_MICRO");
        int selected = !mode ? 0 : !strcmp(mode,"prefill") ? 1 : !strcmp(mode,"decode") ? 2 : !strcmp(mode,"pair") ? 3 : !strcmp(mode,"conv") ? 4 : !strcmp(mode,"all") ? 7 : -1;
        if (selected < 0) return 2;
        scratch_select(selected);
        if (!micro_compile()) return 1;
        int rc = micro_bench_main(argc,argv);
        fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 prefill_pair=%llu decode_pair=%llu conv_norm=%llu\n",
            (unsigned long long)micro_prefill_calls,(unsigned long long)micro_decode_calls,
            (unsigned long long)micro_conv_calls);
        return rc;
    }
}
