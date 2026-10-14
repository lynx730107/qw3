/* Scheduling configured before the original warmup; original benchmark timers. */
#define main projection_barriers_original_bench_main
#include "../../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *mode=getenv("QW3_PRIVATE_PROJECTION_BARRIERS");
        if (!mode || (strcmp(mode,"0") && strcmp(mode,"1"))) return 2;
        private_ffn_enabled=atoi(mode);
        int rc=projection_barriers_original_bench_main(argc,argv);
        if (private_error || private_decode_active || private_ffn_active || g_batch_concurrent) return 1;
        fprintf(stderr,"scope_stats two_windows=%llu four_windows=%llu\n",(unsigned long long)private_two_windows,(unsigned long long)private_four_windows);
        fprintf(stderr,"dispatch_stats dispatches=%llu barriers=%llu independent_skips=%llu\n",(unsigned long long)private_dispatches,(unsigned long long)private_barriers,(unsigned long long)private_skips);
        fprintf(stderr,"PASS projection_barriers_native variant=%d calls=%llu additional_weight_buffer_bytes=0\n",
                private_ffn_enabled,(unsigned long long)private_ffn_calls);
        return rc;
    }
}
