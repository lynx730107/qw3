"""Build experiments in a fresh directory using this checkout's runtime and archived probes."""
from pathlib import Path
import json
import hashlib
import shutil
import subprocess
import sys

archive = Path(__file__).resolve().parent
root = archive.parents[2]
work = Path(sys.argv[1]).resolve()
work.mkdir(exist_ok=False)
old = archive.parent / 'staged128_partial_2026-11-17'
for path in [*root.glob('qw3*.h'),root/'qw3.c',root/'qw3_metal.m',root/'qw3_bench.c']:
    shutil.copy2(path, work/path.name)
shutil.copytree(root/'metal', work/'metal')
shutil.copytree(root/'tests',work/'tests',ignore=shutil.ignore_patterns('.*','test-vectors'))
shutil.copytree(root/'tests/.gqa_heads_probe', work/'tests/.gqa_heads_probe')
shutil.copytree(archive.parent/'iq3_nibble_2026-10-14/source_snapshot/tests/.iq3_nibble',work/'tests/.iq3_nibble')

flags = ['cc','-O3','-ffast-math','-mcpu=native','-Wall','-Wextra']
objc = [*flags,'-fno-finite-math-only','-fobjc-arc','-I.','-Itests']
libs = ['-lm','-pthread','-framework','Foundation','-framework','Metal']
def build(args,cwd=work):
    subprocess.run(args,cwd=cwd,check=True)
build([*flags,'-std=c99','-c','qw3.c','-o','qw3.o'])
build(['patch','-o','matched_bench.c','qw3_bench.c',str(archive/'warmup.diff')])
build([*flags,'-std=c99','-c','matched_bench.c','-o','matched_bench.o'])
build([*flags,'-fobjc-arc','qw3_metal.m','qw3.o','matched_bench.o',*libs,'-o','matched_bench'])

gen=(old/'prepare_partial.py').read_text()
assert gen.count('n_tokens >= 96u')==1
(work/'prepare.py').write_text(gen.replace('n_tokens >= 96u','n_tokens >= 64u'))
partial=work/'partial'
build(['python3',str(work/'prepare.py'),str(work),str(partial),'--all-formats'])
for f in ('gqa_heads_numeric_oracles.inc','optimization_acceptance.h'):
    shutil.copy2(work/'tests'/f,partial/f)
src=(old/'quality.m').read_text()
src=src.replace('{96, 97, 127, 128, 129, 159, 224}', '{63, 64, 65, 69, 71, 87, 95, 96, 127, 128, 193, 223}')
src=src.replace('argc == 4 ? 7 : 1', 'argc == 4 ? (int)(sizeof(lengths)/sizeof(*lengths)) : 1')
(partial/'quality.m').write_text(src)
build([*objc,'quality.m',*libs,'-o','quality'],partial)
shutil.copy2(archive/'turn.m',partial/'turn.m')
build([*objc,'turn.m','qw3.o',*libs,'-o','turn'],partial)
for kind in ('native','model'):
    src = f'tests/bench_gqa_heads_probe_native.m' if kind=='native' else 'tests/test_gqa_heads_probe_model.m'
    build([*objc,src,*libs,'-o','gqa_'+kind])

matrix=work/'tests/test_iq3_nibble.m'
src='#include "optimization_acceptance.h"\n'+matrix.read_text()
a=src.index('else if (memcmp(reference.bytes,y,output.length))')
b=src.index('\n                }',a)
src=src[:a]+'''else {
                        qw3_opt_error error={0};
                        const float *a=reference.bytes;
                        for (unsigned i=5;i<no+5u;i++) qw3_opt_error_add(&error,a[i],y[i]);
                        printf("metrics nmse=%.12g max_abs=%.12g bits=%llu\\n",
                            qw3_opt_error_nmse(&error),error.max_abs,(unsigned long long)error.different_bits);
                        if (!qw3_opt_kernel_valid(&error)) return 1;
                    }'''+src[b:]
matrix.write_text(src.replace('matrix_exact','matrix_numeric').replace('finite_bit_exact=all','finite_numeric=all'))
model=work/'tests/.iq3_nibble/model.inc'
src=model.read_text()
src='#include "../gqa_heads_numeric_oracles.inc"\n'+src[src.index('int main('):]
src=src.replace('|| expected!=token) goto done;',') goto done;\n                if (phase) token=expected;')
model.write_text(src.replace('bit_exact canonical_state=exact','numeric canonical_state=numeric').replace('        rc=0;', '        grid_numeric_report();rc=0;'))
for name,src in [('matrix','tests/test_iq3_nibble.m'),('model','tests/test_iq3_nibble_model.m'),('native','tests/bench_iq3_nibble_native.m')]:
    build([*objc,src,*libs,'-o','nibble_'+name])
build(['python3','tests/prepare_decode_profile.py'])
src=(work/'tests/test_decode_profile.m').read_text()
start=src.index('int main(')
src=src[:start].replace('#include ".decode_profile/oracle.inc"','')+(archive/'profile.inc').read_text()
(work/'tests/profile_depth.m').write_text(src)
build([*objc,'tests/profile_depth.m',*libs,'-o','profile_depth'])
paths=[work/n for n in ('matched_bench','gqa_model','partial/trace','gqa_native','nibble_native','nibble_model','nibble_matrix','profile_depth','partial/turn','partial/quality','partial/bench','partial/qw3_metal.m')]
(archive/'diagnostic-hashes.json').write_text(json.dumps({p.relative_to(work).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
