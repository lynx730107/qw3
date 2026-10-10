#include "model_residency_candidates.inc"
#define main residency_original_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    int rc=residency_original_bench_main(argc,argv);
    fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 residency_requests=%llu residency_views=%llu residency_bytes=%llu\n",
        (unsigned long long)residency_requests,(unsigned long long)residency_views,(unsigned long long)residency_bytes);
    return rc;
}
