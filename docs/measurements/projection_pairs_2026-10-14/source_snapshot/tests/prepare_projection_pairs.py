"""Diagnostic AB/BA order in the unchanged native timing loop, one engine/session."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.projection_pairs'
out.mkdir(exist_ok=True)
base = root / 'docs/measurements/decode_projection_barriers_2026-10-14'
m = json.loads((base / 'manifest.json').read_text())
assert m['all_passed']


def frozen(name):
    key = 'tests/.decode_projection_barriers/' + name
    data = (base / m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m['source_sha256'][key]
    return data


for name in ('hooks.inc', 'qw3_private.c', 'qw3_private_metal.m'):
    (out / name).write_bytes(frozen(name))
source = (root / 'qw3_bench.c').read_text()
start = '    for (int rep = 0; !rc && rep < cfg->repetitions; rep++) {\n'
assert source.count(start) == 1
before = '''        if (cfg->repetitions!=12 || n_prompt!=0 || n_gen!=32 || cfg->depth!=128 || cfg->ctx_alloc!=256) {rc=1;break;}
        static const int order[12]={0,1,1,0,0,1,1,0,0,1,1,0};
        private_ffn_enabled=order[rep];
        private_ffn_calls=0;private_dispatches=0;private_barriers=0;private_skips=0;
        private_two_windows=0;private_four_windows=0;
'''
source = source.replace(start, start + before)
marker = '        samples_sec[rep] = t1 - t0;'
assert source.count(marker) == 1
after = '''        if (private_error || private_decode_active || private_ffn_active || g_batch_concurrent ||
            private_ffn_calls!=(private_ffn_enabled ? 2528u : 0u) ||
            private_two_windows!=(private_ffn_enabled ? 992u : 0u) ||
            private_four_windows!=(private_ffn_enabled ? 1536u : 0u) ||
            private_dispatches!=(private_ffn_enabled ? 31296u : 0u) ||
            private_barriers!=(private_ffn_enabled ? 25696u : 0u) ||
            private_skips!=(private_ffn_enabled ? 5600u : 0u)) {rc=1;break;}
        private_pair_runs++;
        fprintf(stderr,"paired_sample repetition=%d variant=%d calls=%llu dispatches=%llu barriers=%llu independent_skips=%llu sessions=1\\n",rep+1,private_ffn_enabled,(unsigned long long)private_ffn_calls,(unsigned long long)private_dispatches,(unsigned long long)private_barriers,(unsigned long long)private_skips);
'''
source = source.replace(marker, after + marker)
assert source.replace(start + before, start).replace(after + marker, marker) == (root / 'qw3_bench.c').read_text()
(out / 'bench_private.c').write_text(source)
(out / 'native.inc').write_text('''/* Diagnostic: original benchmark timer and work; hook is outside timer. */
static unsigned private_pair_runs;
#define main projection_pairs_original_main
#include "bench_private.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        private_ffn_enabled=0;
        int rc=projection_pairs_original_main(argc,argv);
        if (!rc && private_pair_runs!=12u) return 1;
        fprintf(stderr,"PASS projection_pairs runs=%u sessions=1 additional_weight_buffer_bytes=0\\n",private_pair_runs);
        return rc;
    }
}
''')
print('AB/BA diagnostic prepared; original timing and reset/prefill loops preserved')
