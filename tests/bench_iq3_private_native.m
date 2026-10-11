#define main iq3_private_quality_main
#include ".iq3_private/quality.inc"
#undef main
static int iq3_private_native_enabled;
static int iq3_private_native_guards_ok=1;
static int iq3_private_native_open(qw3_engine **engine,const qw3_engine_options *options) {
    int rc=qw3_engine_open(engine,options);
    if (rc) return rc;
    if (!iq3_private_compile() || (iq3_private_native_enabled && !iq3_private_prepare(*engine))) {
        qw3_engine_close(*engine);*engine=NULL;return 1;
    }
    scratch_select(iq3_private_native_enabled);
    return 0;
}
static void iq3_private_native_close(qw3_engine *engine) {
    if (iq3_private_native_enabled && iq3_private_weights)
        for (id<MTLBuffer> buffer in iq3_private_weights.allValues)
            if (!iq3_private_guards(buffer)) iq3_private_native_guards_ok=0;
    qw3_engine_close(engine);
}
#define qw3_engine_open iq3_private_native_open
#define qw3_engine_close iq3_private_native_close
#define main iq3_private_native_bench_main
#include "../qw3_bench.c"
#undef main
#undef qw3_engine_open
#undef qw3_engine_close
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *enabled=getenv("QW3_METAL_IQ3_PRIVATE");
        if (!enabled || (strcmp(enabled,"0") && strcmp(enabled,"1"))) return 2;
        iq3_private_native_enabled=!strcmp(enabled,"1");
        int rc=iq3_private_native_bench_main(argc,argv);
        if (!rc) {
            const uint64_t calls=iq3_private_calls[0]+iq3_private_calls[1];
            if (iq3_private_native_enabled) {
                if (!calls || !iq3_private_weights || iq3_private_redirects!=iq3_private_calls[0]+2u*iq3_private_calls[1]) return 1;
                if (!iq3_private_native_guards_ok) return 1;
            } else if (calls || iq3_private_redirects || iq3_private_weights) return 1;
            fprintf(stderr,"PASS iq3_private_native enabled=%d calls=%llu redirects=%llu bytes=%llu\n",
                iq3_private_native_enabled,(unsigned long long)calls,(unsigned long long)iq3_private_redirects,(unsigned long long)iq3_private_bytes);
        }
        return rc;
    }
}
