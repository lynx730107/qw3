"""Verify IQ3_XXS sign-only changes and raw finite bit-exact evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index',action='store_true')
args=parser.parse_args()
archive=Path(__file__).resolve().parent
root=archive.parents[2]
m=json.loads((archive/'manifest.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
for name,h in m['source_sha256'].items():
    frozen=(archive/m['source_snapshot_paths'][name]).resolve()
    assert sha(root/name)==sha(frozen)==h,name
    if args.index:
        content=subprocess.run(['git','show',':'+str(frozen.relative_to(root))],cwd=root,capture_output=True,check=True).stdout
        assert hashlib.sha256(content).hexdigest()==h,name
for name,h in {**m['binary_sha256'],**m['production_binary_sha256']}.items():assert sha(root/name)==h,name
assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged','store_unchanged','all_passed'))
assert m['completed_utc'] and len(m['runs'])==3 and all(r['exit_code']==0 for r in m['runs'])
assert [r['name'] for r in m['runs']]==['matrix_v1','matrix_v2','model']
assert not (archive/'build.stderr').read_text()
model=Path(m['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
for variant in (1,2):
    out=(archive/f'matrix_v{variant}.stdout').read_text()
    assert len(re.findall(r'^matrix_exact .*native_threads=64 candidate_threads=64 guards=exact input=unchanged$',out,re.M))==192
    assert f'PASS iq3xxs_sign={variant} matrix_cases=192 finite_bit_exact=all' in out
    log=(archive/f'matrix_v{variant}.stderr').read_text();assert 'Metal API Validation Enabled' in log
    for v,size in ((0,4224),(1,4096),(2,4096)):
        assert f'iq3xxs_sign_metadata variant={v} width=32 max_threads=1024 static_threadgroup_bytes={size} dynamic_threadgroup_bytes=128 threads=64' in log
out=(archive/'model.stdout').read_text()
assert re.findall(r'^PASS iq3xxs_sign_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$',out,re.M)==[('0','0'),('1','1088'),('2','1088')]
core=(root/'qw3.c').read_text()
table=re.search(r'const uint8_t qw3_ksigns_iq2xs\[128\] = \{(.*?)\};',core,re.S)
values=[int(v) for v in re.findall(r'\d+',table[1])]
assert values==[i|((bin(i).count('1')&1)<<7) for i in range(128)]
source=(root/'metal/qw3_core_linear.metal').read_text()
def extract(text,name):
    a=text.index(name);return text[a:text.index('\n}\n',a)+3]
helper=extract(source,'inline float qw3_iq3xxs_dot32_vector(')
kernel=extract(source,'kernel void qw3_matvec_iq3_xxs_fast(')
shader=(root/'tests/.iq3xxs_sign/kernels.metal').read_text()
for variant in (1,2):
    h=extract(shader,f'inline float private_iq3xxs_sign_dot{variant}(')
    h=h.replace(f'private_iq3xxs_sign_dot{variant}','qw3_iq3xxs_dot32_vector').replace('device const uchar *signs','threadgroup const uchar *signs')
    h=h.replace('''        const uint sign_index = (packed >> (7u * l)) & 127u;
        const uint s = sign_index | ((popcount(sign_index) & 1u) << 7u);''','        const uint s = uint(signs[(packed >> (7u * l)) & 127u]);')
    assert h==helper
    k=extract(shader,f'kernel void private_iq3xxs_sign_fast{variant}(')
    k=k.replace(f'private_iq3xxs_sign_fast{variant}','qw3_matvec_iq3_xxs_fast').replace(f'private_iq3xxs_sign_dot{variant}','qw3_iq3xxs_dot32_vector')
    k=k.replace('    device const uchar *signs = sign_table;','    threadgroup uchar signs[128];')
    k=k.replace('    threadgroup_barrier(mem_flags::mem_threadgroup);','    for (uint i = tid; i < 128u; i += 64u) signs[i] = sign_table[i];\n    threadgroup_barrier(mem_flags::mem_threadgroup);')
    assert k==kernel
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'guarded_matrix_cases':384,'model_phases':3,'candidate_dispatches_per_continuation':1088,
        'single_reused_ctx256_session':True,'all_original_sign_entries_match_parity':128,
        'floating_point_expressions_loops_reductions_preserved':True,
        'static_threadgroup_bytes':[4224,4096,4096],'fixture_dynamic_threadgroup_bytes':128,
        'additional_weight_or_session_data_buffers':0,'production_unchanged_during_gate':True,
        'scope':'private prefix128/greedy32 gate; extended prefix/checkpoint gate required before adoption'}
(archive/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
