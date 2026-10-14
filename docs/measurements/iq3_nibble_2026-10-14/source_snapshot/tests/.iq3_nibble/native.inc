/* Ordinary timing runner: compile before warmup, retain original buffers. */
static int grid_native_open(qw3_engine **e,const qw3_engine_options *o) {
    int rc=qw3_engine_open(e,o);if (rc) return rc;
    if (!iq3_nibble_compile()) {qw3_engine_close(*e);*e=NULL;return 1;}
    scratch_select(nibble_variant!=0);return 0;
}
#define qw3_engine_open grid_native_open
#define main grid_native_bench_main
#include "../../qw3_bench.c"
#undef main
#undef qw3_engine_open
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *mode=getenv("QW3_PRIVATE_IQ3_NIBBLE");
        if (!mode || (strcmp(mode,"0") && strcmp(mode,"1"))) return 2;
        nibble_variant=atoi(mode);int rc=grid_native_bench_main(argc,argv);
        if (!rc && ((nibble_variant && !iq3_nibble_calls) || (!nibble_variant && iq3_nibble_calls))) return 1;
        fprintf(stderr,"PASS iq3_nibble_native variant=%d calls=%llu additional_weight_buffer_bytes=0\n",
            nibble_variant,(unsigned long long)iq3_nibble_calls);return rc;
    }
}
