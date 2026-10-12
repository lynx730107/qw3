"""Verify immutable inputs, numerical coverage and every native timing sample."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--index',action='store_true')
args=parser.parse_args()
base=Path(__file__).resolve().parent
root=base.parents[2]
m=json.loads((base/'manifest.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for name,expected in m['source_sha256'].items():
    frozen=(base/m['source_snapshot_paths'][name]).resolve()
    assert sha(frozen)==expected and sha(root/name)==expected, name
    if args.index:
        data=subprocess.check_output(['git','show',':'+str(frozen.relative_to(root))],cwd=root)
        assert hashlib.sha256(data).hexdigest()==expected, ('index',name)
for name,expected in {**m['binary_sha256'],**m['production_binary_sha256']}.items():
    assert sha(root/name)==expected,name
model=Path(m['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged','store_unchanged','all_passed'))
assert not (base/'build.stderr').read_text()
assert len(m['runs'])==3 and all(r['exit_code']==0 for r in m['runs'])
for v in (1,2):
    output=(base/f'matrix_v{v}.stdout').read_text()
    assert len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$',output,re.M))==36
    assert output.count('input_offset=4 guards=')==18 and output.count('input_offset=16 guards=')==18
    assert f'PASS iq3_input={v} matrix_cases=36 finite_bit_exact=all' in output
    metadata=(base/f'matrix_v{v}.stderr').read_text()
    assert len(re.findall(r'^iq3_input_metadata .*width=32 max_threads=1024 static_threadgroup_bytes=1024 dynamic_threadgroup_bytes=256 threads=64$',metadata,re.M))==3
output=(base/'model.stdout').read_text()
assert len(re.findall(r'^PASS iq3_input_model .*finite_logits=bit_exact canonical_state=exact history=exact .*sessions=1 additional_weight_buffer_bytes=0$',output,re.M))==3
assert 'phase=0' in output and 'calls=0' in output and output.count('calls=2336')==2
# Undo only the activation loads and the optional integer-loop unroll hint.
generator=ast.parse((root/'tests/prepare_iq3_input.py').read_text())
values={n.targets[0].id:ast.literal_eval(n.value) for n in generator.body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('old','new')}
source=(root/'metal/qw3_core_moe.metal').read_text()
name='qw3_matvec_iq3_s_fast'
a=source.index('kernel void '+name+'(')
original=source[a:source.index('\n}\n',a)+3]
shader=(root/'tests/.iq3_input/kernels.metal').read_text()
for v in (1,2):
    private=f'private_iq3_input_fast{v}'
    a=shader.index('kernel void '+private+'(')
    body=shader[a:shader.index('\n}\n',a)+3]
    assert body.replace(private,name).replace(values['new'],values['old']).replace('#pragma unroll(2)\n','')==original
    assert body.count('threadgroup_barrier(')==original.count('threadgroup_barrier(')==1
native=json.loads((base/'native/manifest.json').read_text())
assert native['order']==[0,1,0,2,0] and len(native['runs'])==5
assert native['source_sha256']==m['source_sha256']
assert native['production_binary_sha256']==m['production_binary_sha256']
assert all(native[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
for i,run in enumerate(native['runs']):
    assert run['exit_code']==0 and run['variant']==native['order'][i]
    assert len(run['samples'])==3 and run['candidate_calls']==(7081 if run['variant'] else 0)
    assert run['environment'].get('MTL_DEBUG_LAYER') is None
    log=(base/'native'/(run['name']+'.stderr')).read_text()
    samples=[{'repetition':int(n),'tps':float(t),'ms':float(ms)} for n,t,ms in
             re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms',log)]
    assert samples==run['samples']
    assert f"PASS iq3_input_native variant={run['variant']} calls={7081 if run['variant'] else 0} additional_weight_buffer_bytes=0" in log
    rows=[row for row in (base/'native'/(run['name']+'.stdout')).read_text().splitlines() if row.startswith('tg,')]
    assert len(rows)==1 and int(rows[0].split(',')[-1])==176761680
analysis=json.loads((base/'native/analysis.json').read_text())
assert analysis['samples_retained']==15
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'guarded_matrix_cases_passed':72,'input_offsets_checked':[4,16],
        'one_reused_session_model_phases':3,'ordinary_samples_verified':15,
        'production_unchanged':True,'additional_weight_and_session_data_buffers':0,
        'scope':'private screen; no eight-prefix/checkpoint gate and no fresh llama comparison',
        'analysis':analysis}
(base/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
