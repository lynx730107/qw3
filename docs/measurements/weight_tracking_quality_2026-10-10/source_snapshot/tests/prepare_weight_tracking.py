"""Generate private read-only GGUF view validation from the native oracle."""
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'tests/.weight_tracking';out.mkdir(exist_ok=True)
quality=(root/'tests/test_native_prefill.m').read_text()
quality=quality.replace('#include "native_prefill_validation.inc"','''#include "../weight_tracking_candidates.inc"
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    setenv("QW3_METAL_DENSE_GDN_CACHE","1",1);
    if (!weight_select(mode)) abort();
}''')
start=quality.index('        if (((mode & 1)');end=quality.index('        rc=0;',start)
quality=quality[:start]+'''        if (!weight_switches[0] || !weight_switches[1] || !weight_view_count ||
            weight_temp[0].count || weight_temp[1].count) goto done;
        printf("PASS weight_tracking tracked_switches=%llu untracked_switches=%llu views=%llu bytes=%llu temp_views=0\\n",
            (unsigned long long)weight_switches[0],(unsigned long long)weight_switches[1],
            (unsigned long long)weight_view_count,(unsigned long long)weight_view_bytes);
'''+quality[end:]
(out/'quality.inc').write_text(quality)
