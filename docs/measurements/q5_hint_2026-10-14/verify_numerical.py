"""Verify original-shader descriptor variants against raw numerical evidence."""
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
for name,expected in m['source_sha256'].items():
    frozen=(archive/m['source_snapshot_paths'][name]).resolve()
    assert sha(root/name)==sha(frozen)==expected,name
    if args.index:
        content=subprocess.run(['git','show',':'+str(frozen.relative_to(root))],cwd=root,capture_output=True,check=True).stdout
        assert hashlib.sha256(content).hexdigest()==expected,name
for name,expected in {**m['binary_sha256'],**m['production_binary_sha256']}.items():
    assert sha(root/name)==expected,name
assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged','store_unchanged','all_passed'))
assert m['completed_utc'] and len(m['runs'])==3 and all(r['exit_code']==0 for r in m['runs'])
assert [r['name'] for r in m['runs']]==['matrix_v1','matrix_v2','model']
assert not (archive/'build.stderr').read_text()
model=Path(m['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
for variant in (1,2):
    out=(archive/f'matrix_v{variant}.stdout').read_text()
    rows=re.findall(r'^matrix_exact .*native_threads=64 candidate_threads=64 guards=exact input=unchanged$',out,re.M)
    assert len(rows)==192
    assert f'PASS q5_hint={variant} matrix_cases=192 finite_bit_exact=all' in out
    for offset in (0,2,4,8):assert out.count(f'weight_offset={offset} row_padding=')==48
    log=(archive/f'matrix_v{variant}.stderr').read_text()
    assert 'Metal API Validation Enabled' in log
    for v,max_threads in ((0,1024),(1,64),(2,64)):
        assert f'q5_hint_metadata variant={v} width=32 max_threads={max_threads} static_threadgroup_bytes=0 dynamic_threadgroup_bytes=128 threads=64' in log
out=(archive/'model.stdout').read_text()
rows=re.findall(r'^PASS q5_hint_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$',out,re.M)
assert rows==[('0','0'),('1','832'),('2','832')]
hooks=(root/'tests/.q5_hint/hooks.inc').read_text()
assert '[g_library newFunctionWithName:@"qw3_matvec_q5_k_fast"]' in hooks
assert 'newLibraryWithSource' not in hooks and 'kernels.metal' not in hooks
assert 'descriptor.maxTotalThreadsPerThreadgroup=64u;' in hooks
assert 'descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=(v==2);' in hooks
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'guarded_matrix_cases':384,'weight_offsets':[0,2,4,8],
        'model_phases':3,'candidate_dispatches_per_continuation':832,
        'single_reused_ctx256_session':True,'original_function_from_original_library':True,
        'descriptor_only_changes':True,'native_and_candidate_threads':64,
        'static_threadgroup_bytes':[0,0,0],'fixture_dynamic_threadgroup_bytes':128,
        'additional_weight_or_session_data_buffers':0,'production_unchanged_during_gate':True,
        'scope':'private prefix128/greedy32 gate; extended prefix/checkpoint gate required before adoption'}
(archive/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
