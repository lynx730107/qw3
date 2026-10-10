"""Generate the private full-state residency control from the native oracle."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.model_residency'
out.mkdir(exist_ok=True)
quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '''#include "../model_residency_candidates.inc"
static void scratch_select(int mode) {
    if (!residency_select(mode != 0)) abort();
}''')
quality = quality.replace('        qw3_engine *e=NULL;',
                          '        setenv("QW3_PRIVATE_MODEL_RESIDENCY","1",1);\n        qw3_engine *e=NULL;')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (!model_residency_set || !residency_active || residency_requests<2 ||
            !residency_views || !residency_bytes) goto done;
        printf("PASS residency_requests=%llu views=%llu bytes=%llu\\n",
            (unsigned long long)residency_requests,(unsigned long long)residency_views,
            (unsigned long long)residency_bytes);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)
