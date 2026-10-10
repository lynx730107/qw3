"""Private session-buffer tracking probe; production source is unchanged."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.hazard_tracking'
out.mkdir(exist_ok=True)
core = (root / 'qw3_metal.m').read_text()
old = '''    return [g_device newBufferWithLength:(NSUInteger)qw3_metal_session_align(bytes)
                                 options:MTLResourceStorageModePrivate];'''
new = '''    id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)qw3_metal_session_align(bytes)
        options:MTLResourceStorageModePrivate |
            (hazard_untracked ? MTLResourceHazardTrackingModeUntracked : 0)];
    if (buffer) {
        MTLHazardTrackingMode expected = hazard_untracked ?
            MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked;
        if (buffer.hazardTrackingMode != expected) abort();
        hazard_allocations[hazard_untracked != 0]++;
        hazard_bytes[hazard_untracked != 0] += buffer.length;
    }
    return buffer;'''
assert core.count(old) == 1
(out / 'core.inc').write_text(core.replace(old, new))
validation = (root / 'tests/native_prefill_validation.inc').read_text()
validation = validation.replace('#include "../qw3_metal.m"', '''#include <stdint.h>
static int hazard_untracked;
static uint64_t hazard_allocations[2], hazard_bytes[2];
#include "core.inc"''')
validation = validation.replace('#include "../qw3.c"', '#include "../../qw3.c"')
start = validation.index('static void scratch_select(int mode) {')
end = validation.index('\n#define qw3_metal_session_batch_dense', start)
validation = validation[:start] + '''static void scratch_select(int mode) {
    scratch_mode = mode;
    hazard_untracked = mode != 0;
    setenv("QW3_METAL_DENSE_PREFILL_FAST", "1", 1);
    setenv("QW3_METAL_DENSE_GDN_CACHE", "1", 1);
}
''' + validation[end:]
(out / 'validation.inc').write_text(validation)
quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('"native_prefill_validation.inc"', '"validation.inc"')
old = '        for (int v=0;v<2;v++) if (qw3_session_create(&s[v],e,512) || s[v]->metal_n_gpu_layers!=64) goto done;'
new = '''        for (int v=0;v<2;v++) {
            scratch_select(v);
            if (qw3_session_create(&s[v],e,512) || s[v]->metal_n_gpu_layers!=64) goto done;
            QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s[v]->metal->obj;
            if (obj.x0.hazardTrackingMode != (v ? MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked)) goto done;
        }'''
assert quality.count(old) == 1
quality = quality.replace(old, new)
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (!hazard_allocations[0] || !hazard_allocations[1]) goto done;
        printf("PASS hazard_tracking tracked_allocations=%llu untracked_allocations=%llu tracked_bytes=%llu untracked_bytes=%llu\\n",
            (unsigned long long)hazard_allocations[0],(unsigned long long)hazard_allocations[1],
            (unsigned long long)hazard_bytes[0],(unsigned long long)hazard_bytes[1]);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)
