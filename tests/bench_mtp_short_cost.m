/* Private paired cost control: same precise recurrence on both batch sides.
 * All-row correctness is independently checked by test-mtp-f32.
 * This cost probe returns only the last head, not speculative throughput. */
#define QW3_MTP_DECODE_ENTRYPOINT qw3_mtp_decoder_probe_entrypoint
#include "bench_mtp_decode.m"

static int short_cost_init(void) {
    const char *mask = getenv("QW3_MTP_SHORT_VECTOR");
    if (mask && decode_int(mask,0,3) < 0) return 0;
    printf("short_cost f32=1 candidate_vector_mask=%d cached_gdn=%d not_speculative_decode=1\n",
        mask ? decode_int(mask,0,3) : 3, getenv("QW3_MTP_GDN_CACHED") != NULL);
    decode_cached_gdn = getenv("QW3_MTP_GDN_CACHED") != NULL;
    return qw3_metal_compile_kernels() && decode_short_init();
}

static void short_cost_select(int variant) {
    decode_short_f32 = 1;
    const char *mask = getenv("QW3_MTP_SHORT_VECTOR");
    decode_short_vector = variant ? (mask ? decode_int(mask,0,3) : 3) : 0;
    decode_cached_gdn = variant && getenv("QW3_MTP_GDN_CACHED") != NULL;
}

#define QW3_BLOCK_CORE_INCLUDED
#define QW3_BLOCK_INIT short_cost_init
#define QW3_BLOCK_SELECT short_cost_select
#include "bench_verify_blocks.c"
