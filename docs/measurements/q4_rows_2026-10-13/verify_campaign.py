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
assert len(m['runs'])==2 and all(r['exit_code']==0 for r in m['runs'])
output=(base/'matrix.stdout').read_text()
assert len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$',output,re.M))==24
assert output.count('candidate_threads=64 guards=')==21
assert 'PASS q4_rows matrix_cases=24 finite_bit_exact=all' in output
assert 'calls=18 fallbacks=6' in (base/'matrix.stderr').read_text()
output=(base/'model.stdout').read_text()
assert len(re.findall(r'^PASS q4_rows_model .*finite_logits=bit_exact canonical_state=exact history=exact .*sessions=1 additional_weight_buffer_bytes=0$',output,re.M))==2
assert 'phase=0' in output and 'calls=0' in output and 'phase=1' in output and 'calls=1088' in output
log=(base/'model.stderr').read_text()
assert 'calls=0 fallbacks=0' in log and 'calls=1088 fallbacks=64' in log
# Undo only integer row mapping, inactive-row guards and zero-lane selection.
tree=ast.parse((root/'tests/prepare_q4_rows.py').read_text())
node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='changes')
changes={('qw3_matvec_q4_k' if isinstance(k,ast.Name) and k.id=='name' else ast.literal_eval(k)):ast.literal_eval(v) for k,v in zip(node.keys,node.values)}
shader=(root/'tests/.q4_rows/kernels.metal').read_text()
restored=shader
for before,after in reversed(list(changes.items())): restored=restored.replace(after,before)
source=(root/'metal/qw3_core_linear.metal').read_text()
a=source.index('kernel void qw3_matvec_q4_k(')
original=source[a:source.index('\n}\n',a)+3]
assert restored==original and shader.count('simd_sum(')==original.count('simd_sum(')==2
assert shader.count('threadgroup_barrier(')==1
native=json.loads((base/'native/manifest.json').read_text())
assert native['order']==[0,1,0,1,0,1,0] and len(native['runs'])==7
assert all(native['source_sha256'][p]==h for p,h in m['source_sha256'].items())
assert len(native['source_sha256'])==len(m['source_sha256'])+1
assert native['cooldown_seconds_by_process']==[180,60,60,60,60,60,60]
for name,expected in native['source_sha256'].items():
    frozen=(base/'native'/native['source_snapshot_paths'][name]).resolve()
    assert sha(frozen)==expected and sha(root/name)==expected,name
    if args.index:
        data=subprocess.check_output(['git','show',':'+str(frozen.relative_to(root))],cwd=root)
        assert hashlib.sha256(data).hexdigest()==expected,('native index',name)
assert native['production_binary_sha256']==m['production_binary_sha256']
assert all(native[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
for i,run in enumerate(native['runs']):
    assert run['exit_code']==0 and run['variant']==native['order'][i]
    assert len(run['samples'])==3 and run['candidate_calls']==(3298 if run['variant'] else 0)
    assert run['environment'].get('MTL_DEBUG_LAYER') is None
    log=(base/'native'/(run['name']+'.stderr')).read_text()
    samples=[{'repetition':int(n),'tps':float(t),'ms':float(ms)} for n,t,ms in
             re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms',log)]
    assert samples==run['samples']
    counts=re.search(r'q4_rows_dispatch_counts calls=(\d+) fallbacks=(\d+)',log)
    assert counts and tuple(map(int,counts.groups()))==((3298,194) if run['variant'] else (0,0))
    assert f"PASS q4_rows_native variant={run['variant']} calls={3298 if run['variant'] else 0} additional_weight_buffer_bytes=0" in log
    rows=[row for row in (base/'native'/(run['name']+'.stdout')).read_text().splitlines() if row.startswith('tg,')]
    assert len(rows)==1 and int(rows[0].split(',')[-1])==176761680
analysis=json.loads((base/'native/analysis.json').read_text())
assert analysis['samples_retained']==21
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'guarded_matrix_cases_passed':24,'candidate_dispatches_checked':18,'fallbacks_checked':6,
        'one_reused_session_model_phases':2,'ordinary_samples_verified':21,
        'production_unchanged':True,'additional_weight_and_session_data_buffers':0,
        'scope':'private screen; no eight-prefix/checkpoint gate and no fresh llama comparison',
        'analysis':analysis}
(base/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
