/* Diagnostic: original benchmark timer and work; hook is outside timer. */
static unsigned private_pair_runs;
#define main projection_pairs_original_main
#include "bench_private.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        private_ffn_enabled=0;
        int rc=projection_pairs_original_main(argc,argv);
        if (!rc && private_pair_runs!=12u) return 1;
        fprintf(stderr,"PASS projection_pairs runs=%u sessions=1 additional_weight_buffer_bytes=0\n",private_pair_runs);
        return rc;
    }
}
