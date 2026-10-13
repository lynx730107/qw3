/* Ordinary timing runner: compile before warmup, retain original buffers. */
static int grid_native_open(qw3_engine **e,const qw3_engine_options *o) {
    int rc=qw3_engine_open(e,o);if (rc) return rc;
    if (!q4_narrow_compile()) {qw3_engine_close(*e);*e=NULL;return 1;}
    return 0;
}
#define qw3_engine_open grid_native_open
#define main grid_native_bench_main
#include "../../qw3_bench.c"
#undef main
#undef qw3_engine_open
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *mode=getenv("QW3_PRIVATE_Q4_NARROW");
        if (!mode || (strcmp(mode,"0") && strcmp(mode,"1"))) return 2;
        int variant=atoi(mode);scratch_select(variant);int rc=grid_native_bench_main(argc,argv);
        if (!rc && ((variant && !q4_narrow_calls) || (!variant && q4_narrow_calls))) return 1;
        q4_narrow_print_counts();
        fprintf(stderr,"PASS q4_narrow_native variant=%d calls=%llu additional_weight_buffer_bytes=0\n",
            variant,(unsigned long long)q4_narrow_calls);return rc;
    }
}
