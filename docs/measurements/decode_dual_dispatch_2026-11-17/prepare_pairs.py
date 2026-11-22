"""Apply OFF/ON and ON/OFF inside the original one-session benchmark."""
from pathlib import Path
import subprocess
import sys

work = Path(sys.argv[1]).resolve()
build = work/'paired_build'
build.mkdir(exist_ok=True)
src = (work/'qw3_bench.c').read_text()
anchor = '    uint64_t session_bytes = 0;\n    for (int rep = 0; !rc && rep < cfg->repetitions; rep++) {\n'
warm = '''    /* Warm the candidate and its pipelines outside the timer. */
    if (!rc) {
        dual_mode = paired_mode;
        qw3_session_invalidate(session);
        if (qw3_session_sync(session, &depth_prefix, err, sizeof(err)) ||
            qw3_session_eval(session, synthetic.v[cfg->depth], err, sizeof(err))) rc = 1;
    }
'''
hook = '''        dual_mode = (rep % 4 == 1 || rep % 4 == 2) ? paired_mode : 0;
        dual_same_calls = dual_mixed_calls = 0;
'''
assert src.count(anchor) == 1
src = src.replace(anchor, warm + anchor + hook)
marker = '        samples_sec[rep] = t1 - t0;'
trace = '''        const unsigned same = dual_mode & 1 ? 12u * 32u : 0;
        const unsigned mixed = dual_mode & 2 ? 25u * 32u : 0;
        if (dual_same_calls != same || dual_mixed_calls != mixed) {
            rc = 1;
            break;
        }
        fprintf(stderr, "dual_pair rep=%d mode=%u same=%u mixed=%u\\n",
                rep+1, dual_mode, dual_same_calls, dual_mixed_calls);
'''
assert src.count(marker) == 1
src = src.replace(marker, trace + marker)
header = 'static int run_llama_style_case(qw3_engine *engine, const bench_config *cfg,\n                                int n_prompt, int n_gen, bool print_header) {\n'
assert src.count(header) == 1
validation = '''    if (!cfg || !paired_mode || paired_mode > 3 || n_prompt || n_gen != 32 ||
        cfg->depth != 128 || cfg->ctx_alloc != 256 || cfg->no_warmup ||
        cfg->repetitions != 8) return 2;
'''
src = src.replace(header, header + validation)
assert src.replace(warm, '').replace(hook, '').replace(trace, '').replace(validation, '') == (work/'qw3_bench.c').read_text()
(build/'bench_private.c').write_text(src)
(build/'driver.m').write_text('''
#include "adapter.inc"
static unsigned paired_mode;
#define main bench_main
#include "bench_private.c"
#undef main
int main(int argc, char **argv) {
    const char *value = getenv("QW3_DUAL_DISPATCH");
    paired_mode = value ? (unsigned)atoi(value) : 0;
    return bench_main(argc, argv);
}
''')
subprocess.run(['cc','-O3','-ffast-math','-mcpu=native','-Wall','-Wextra','-fobjc-arc',
                '-I'+str(work),'driver.m','-lm','-pthread','-framework','Foundation','-framework','Metal',
                '-o','paired'],cwd=build,check=True)
