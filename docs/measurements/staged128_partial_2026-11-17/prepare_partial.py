"""Prepare an isolated candidate from the baseline copy used for the screen."""
from pathlib import Path
import re
import shutil
import subprocess
import sys

baseline, work = map(Path, sys.argv[1:3])
all_formats = sys.argv[3:] == ['--all-formats']
work.mkdir()
for name in ('qw3.c', 'qw3.h', 'qw3_metal.h', 'qw3_metal.m', 'qw3_bench.c',
             'qw3_runtime_fingerprint.h', 'qw3.o'):
    shutil.copy2(baseline / name, work / name)
shutil.copytree(baseline / 'metal', work / 'metal')
host = work / 'qw3_metal.m'
source = host.read_text()
anchor = 'static int qw3_metal_staged128_enabled(void) {'
helper = '''/* Diagnostic only. Mode 1 keeps paired gate/up; modes 2/3 stage it. */
static unsigned partial_mode;
#ifdef QW3_PARTIAL_TRACE
static unsigned partial_hist[129][4];
#endif
static int qw3_partial_stage(uint32_t n_tokens) {
    return partial_mode && n_tokens >= 96u && n_tokens < 128u;
}

'''
assert source.count(anchor) == 1
source = source.replace(anchor, helper + anchor)
for fmt, hist in (('iq4_xs', 0), ('iq3_s', 1)):
    needle = ('const int use_staged128 = n_tokens % 128u == 0u &&\n'
              f'            g_matmul_{fmt}_mpp_staged128_pipeline &&')
    assert source.count(needle) == 1
    start = source.index(needle)
    end = source.index('qw3_metal_end_compute_encoder(cb, direct);', start)
    section = source[start:end].replace('n_tokens % 128u == 0u &&',
        '(n_tokens % 128u == 0u || qw3_partial_stage(n_tokens)) &&', 1)
    assert section.count('use_staged128 ? n_tokens / 128u') == 1
    section = section.replace('use_staged128 ? n_tokens / 128u', 'use_staged128 ? (n_tokens + 127u) / 128u')
    section += (f'#ifdef QW3_PARTIAL_TRACE\n'
                f'        if (n_tokens <= 128u) partial_hist[n_tokens][{hist * 2} + use_staged128]++;\n'
                '#endif\n        ')
    source = source[:start] + section + source[end:]
needle = '!(n_tokens % 128u == 0u && g_matmul_iq3_s_mpp_staged128_pipeline &&'
assert source.count(needle) == 1
source = source.replace(needle, '''!((n_tokens % 128u == 0u ||
           (partial_mode >= 2 && qw3_partial_stage(n_tokens))) &&
          g_matmul_iq3_s_mpp_staged128_pipeline &&''')
if all_formats:
    for fmt in ('q3_k', 'q4_q5_k', 'iq2_s', 'iq3_xxs'):
        needle = f'n_tokens % 128u == 0u &&\n            g_matmul_{fmt}_mpp_staged128_pipeline'
        assert source.count(needle) == 1
        source = source.replace(needle, '(n_tokens % 128u == 0u ||\n'
            '             (partial_mode == 3 && qw3_partial_stage(n_tokens))) &&\n'
            f'            g_matmul_{fmt}_mpp_staged128_pipeline')
    for needle in ('MTLSizeMake(use_staged128 ? n_tokens / 128u :',
                   'MTLSizeMake(n_tokens / 128u, (n_out + 63u) / 64u, 1u) : (use_mpp ?'):
        assert needle in source
        source = source.replace(needle, needle.replace('n_tokens / 128u', '(n_tokens + 127u) / 128u'))
host.write_text(source)

def guard_shader(source, name):
    start = source.index(f'kernel void qw3_matmul_{name}(')
    end = source.index('\n}\n', start)
    section = source[start:end]
    assert section.count('    threadgroup half *sa') == 1
    section = section.replace('    threadgroup half *sa', '''    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = token0 < int(args.n_tokens);
    threadgroup half *sa''', 1)
    section, runs = re.subn(r'(?m)^( +)auto mB = tB.slice\(([^,]+),\s*r1 \+ int\(simd_group\) \* ROWS_PER_SG\);\n\1mm.run\(mB, mA, cT\);',
                             r'\1if (token_group_active) {\n\1    auto mB = tB.slice(\2, token0);\n\1    mm.run(mB, mA, cT);\n\1}', section)
    assert runs == 1
    anchor = '    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));'
    assert section.count(anchor) == 1
    section = section.replace(anchor, '    if (token_group_active) cT.store(tD.slice(r0, token0));')
    return source[:start] + section + source[end:]

formats = {'qw3_core_moe.metal': ['iq4_xs_mpp_staged128', 'iq3_s_mpp_staged128',
                                 'iq3_s_mpp_staged128_k64']}
if all_formats:
    formats['qw3_core_linear.metal'] = [f'{fmt}_mpp_staged128' for fmt in
                                        ('q3_k', 'q4_q5_k', 'iq2_s', 'iq3_xxs')]
for filename, kernels in formats.items():
    shader = work / 'metal' / filename
    source = shader.read_text()
    for kernel in kernels:
        source = guard_shader(source, kernel)
    shader.write_text(source)

driver = r'''
#include "qw3_metal.m"
#define main original_bench_main
#include "qw3_bench.c"
#undef main
int main(int argc, char **argv) {
    const char *mode = getenv("QW3_PARTIAL_STAGED");
    if (mode) partial_mode = (unsigned)atoi(mode);
    if (partial_mode > 2) return 2;
    int rc = original_bench_main(argc, argv);
#ifdef QW3_PARTIAL_TRACE
    for (unsigned i = 1; i <= 128; i++) {
        const unsigned *h = partial_hist[i];
        if (h[0] || h[1] || h[2] || h[3])
            fprintf(stderr, "partial_trace batch=%u iq4_direct=%u iq4_staged=%u iq3_direct=%u iq3_staged=%u\n",
                    i, h[0], h[1], h[2], h[3]);
    }
#endif
    return rc;
}
'''
(work / 'bench_driver.m').write_text(driver.replace('partial_mode > 2', 'partial_mode > 3') if all_formats else driver)
flags = ['-O3', '-ffast-math', '-mcpu=native', '-Wall', '-Wextra', '-fobjc-arc']
for name, extra in [('bench', []), ('trace', ['-DQW3_PARTIAL_TRACE'])]:
    subprocess.run(['cc', *flags, *extra, 'bench_driver.m', 'qw3.o', '-lm', '-pthread',
                    '-framework', 'Foundation', '-framework', 'Metal', '-o', name],
                   cwd=work, check=True)
