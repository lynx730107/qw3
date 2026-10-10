#include "combination_candidates.inc"
#define main combo_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *name=getenv("QW3_PRIVATE_COMBINATION");
        int mode=!name ? 0 : !strcmp(name,"residual") ? 1 : !strcmp(name,"round") ? 2 : !strcmp(name,"both") ? 3 : -1;
        if (mode<0) return 2;
        scratch_select(mode);
        if (!combo_compile()) return 1;
        int rc=combo_bench_main(argc,argv);
        fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 residual_norm=%llu round_norm=%llu round_saved=%llu\n",
            combo_residual_calls,combo_round_calls,combo_round_saved);
        return rc;
    }
}
