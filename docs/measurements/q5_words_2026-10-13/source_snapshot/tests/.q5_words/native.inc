/* Ordinary timing runner: compile before warmup, retain original buffers. */
static int grid_native_open(qw3_engine **e,const qw3_engine_options *o) {
    int rc=qw3_engine_open(e,o);if (rc) return rc;
    if (!q5_words_compile()) {qw3_engine_close(*e);*e=NULL;return 1;}
    return 0;
}
#define qw3_engine_open grid_native_open
#define main grid_native_bench_main
#include "../../qw3_bench.c"
#undef main
#undef qw3_engine_open
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *mode=getenv("QW3_PRIVATE_Q5_WORDS");
        if (!mode || (strcmp(mode,"0") && strcmp(mode,"1") && strcmp(mode,"2"))) return 2;
        int variant=atoi(mode);words_variant=variant ? variant : 1;scratch_select(variant);int rc=grid_native_bench_main(argc,argv);
        if (!rc && ((variant && !q5_words_calls) || (!variant && q5_words_calls))) return 1;
        q5_words_print_counts();
        fprintf(stderr,"PASS q5_words_native variant=%d calls=%llu additional_weight_buffer_bytes=0\n",
            variant,(unsigned long long)q5_words_calls);return rc;
    }
}
