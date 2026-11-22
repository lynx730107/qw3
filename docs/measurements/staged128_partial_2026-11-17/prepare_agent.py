"""Build an isolated agent with per-process diagnostic mode and dispatch counters."""
from pathlib import Path
import subprocess
import sys

repo = Path.cwd()
work = Path(sys.argv[1]).resolve()
build = work / 'agent_build'
build.mkdir(exist_ok=True)
build.joinpath('metal_probe.m').write_text(r'''
#define QW3_PARTIAL_TRACE 1
#include "qw3_metal.m"
__attribute__((constructor)) static void agent_projection_mode(void) {
    const char *value = getenv("QW3_PARTIAL_STAGED");
    partial_mode = value ? (unsigned)atoi(value) : 0;
}
__attribute__((destructor)) static void agent_projection_trace(void) {
    unsigned count = 0;
    for (unsigned i = 96; i < 128; i++)
        count += partial_hist[i][1] + partial_hist[i][3];
    const char *trace = getenv("QW3_PARTIAL_AGENT_TRACE");
    if (!trace) return;
    FILE *out = fopen(trace, "a");
    if (out) {
        fprintf(out, "partial_agent_dense_dispatches=%u\n", count);
        fclose(out);
    }
}
''')
flags = ['-O3', '-ffast-math', '-mcpu=native', '-Wall', '-Wextra']
libs = ['-lm', '-pthread', '-framework', 'Foundation', '-framework', 'Metal']
subprocess.run(['cc', *flags, '-fobjc-arc', '-I'+str(work), '-c',
                'metal_probe.m', '-o', 'probe.o'], cwd=build, check=True)
for name in ('qw3_agent', 'linenoise'):
    subprocess.run(['cc', *flags, '-std=c99', '-c', str(repo/f'{name}.c'), '-o',
                    f'{name}.o'], cwd=build, check=True)
subprocess.run(['cc', 'qw3_agent.o', 'linenoise.o', str(work/'qw3.o'), 'probe.o', *libs,
                '-o', 'agent'], cwd=build, check=True)

# Extend this test's prompt from 128+64 to 128+96 to hit the new path on resume.
pressure = (repo/'tests/test_agent_context_pressure.m').read_text()
pressure = pressure.replace('#include "../qw3_metal.m"', '#include "metal_probe.m"')
pressure = pressure.replace('#include "../qw3.c"', '#include "qw3.c"')
pressure = pressure.replace('#include "../qw3_agent.c"', '#include "qw3_agent.c"')
pressure = pressure.replace('192', '224')
(build/'pressure.m').write_text(pressure)
subprocess.run(['cc', *flags, '-fobjc-arc', '-fno-finite-math-only',
                '-I'+str(work), '-I'+str(repo), 'pressure.m', 'linenoise.o', *libs,
                '-o', 'pressure'], cwd=build, check=True)
