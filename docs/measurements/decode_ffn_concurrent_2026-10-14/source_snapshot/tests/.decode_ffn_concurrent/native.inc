/* Scheduling configured before the original warmup; original benchmark timers. */
#define main ffn_concurrent_original_bench_main
#include "../../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *mode=getenv("QW3_PRIVATE_FFN_CONCURRENT");
        if (!mode || (strcmp(mode,"0") && strcmp(mode,"1"))) return 2;
        private_ffn_enabled=atoi(mode);
        int rc=ffn_concurrent_original_bench_main(argc,argv);
        if (private_ffn_active || g_batch_concurrent) return 1;
        fprintf(stderr,"PASS ffn_concurrent_native variant=%d calls=%llu additional_weight_buffer_bytes=0\n",
                private_ffn_enabled,(unsigned long long)private_ffn_calls);
        return rc;
    }
}
