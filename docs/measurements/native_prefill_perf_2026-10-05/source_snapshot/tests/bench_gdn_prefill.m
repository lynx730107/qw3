#include "gdn_prefill_candidates.inc"
#define main probe_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    int rc=probe_bench_main(argc,argv);
    fprintf(stderr,"private dispatches pair=0 add=0 gdn=%llu\n",(unsigned long long)probe_gdn_calls);
    return rc;
}
