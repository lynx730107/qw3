#include ".sum8_candidates/candidates.inc"
#define main sum8_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *name=getenv("QW3_PRIVATE_COMBINATION");
        int mode=!name ? 0 : !strcmp(name,"logits") ? 4 : !strcmp(name,"all") ? 7 : -1;
        if (mode<0) return 2;
        scratch_select(mode);
        if (!sum8_compile()) return 1;
        int rc=sum8_bench_main(argc,argv);
        fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 residual_norm=%llu round_norm=%llu round_saved=%llu sum8_head=%llu\n",
            combo_residual_calls,combo_round_calls,combo_round_saved,sum8_head_calls);
        return rc;
    }
}
