/* Native single-session protocol, with verified allocation and fence counters. */
#include ".hazard_fences/validation.inc"
#define main hazard_fences_original_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    const char *env=getenv("QW3_PRIVATE_SESSION_FENCES");
    int mode=env && !strcmp(env,"1");
    hazard_variant=2;scratch_select(mode);
    int rc=hazard_fences_original_bench_main(argc,argv);
    if (!rc && (!hazard_allocations[mode] || hazard_allocations[!mode] || hazard_barriers ||
        (mode && (!hazard_waits || !hazard_updates)) || (!mode && (hazard_waits || hazard_updates)))) rc=1;
    fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 session_untracked=%d fence_waits=%llu fence_updates=%llu tracked_allocations=%llu untracked_allocations=%llu barriers=%llu\n",
        mode,(unsigned long long)hazard_waits,(unsigned long long)hazard_updates,
        (unsigned long long)hazard_allocations[0],(unsigned long long)hazard_allocations[1],
        (unsigned long long)hazard_barriers);
    return rc;
}
