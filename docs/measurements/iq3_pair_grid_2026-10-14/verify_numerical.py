"""Verify paired IQ3 packed lookup evidence and the precise source changes."""
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
    rows=re.findall(r'^matrix_exact .*guards=exact input=unchanged weights=unchanged$',out,re.M)
    assert len(rows)==96
    assert out.count('input_offset=16 padding=')==out.count('input_offset=32 padding=')==48
    assert f'PASS iq3_pair_grid={variant} matrix_cases=96 finite_bit_exact=all' in out
    log=(archive/f'matrix_v{variant}.stderr').read_text()
    assert 'Metal API Validation Enabled' in log
    for v,size in ((0,8192),(1,1024),(2,0)):
        assert f'iq3_pair_grid_metadata variant={v} width=32 max_threads=1024 static_threadgroup_bytes={size} dynamic_threadgroup_bytes=128 threads=64' in log
out=(archive/'model.stdout').read_text()
rows=re.findall(r'^PASS iq3_pair_grid_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$',out,re.M)
assert rows==[('0','0'),('1','608'),('2','608')]

source=(root/'metal/qw3_core_moe.metal').read_text()
def extract(text,name):
    a=text.index(name)
    return text[a:text.index('\n}\n',a)+3]
original_helper=extract(source,'inline float qw3_iq3s_dot32_vector(')
original_kernel=extract(source,'kernel void qw3_matvec_iq3_s_pair_fast(')
shader=(root/'tests/.iq3_pair_grid/kernels.metal').read_text()
original_grid='''    threadgroup float4 grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = qw3_iq3s_float_grid(kgrid[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);'''
for variant in (1,2):
    helper=extract(shader,f'inline float private_iq3_pair_grid_dot{variant}(')
    helper=helper.replace(f'private_iq3_pair_grid_dot{variant}','qw3_iq3s_dot32_vector')
    helper=helper.replace(('threadgroup' if variant==1 else 'device')+' const ushort *grid','threadgroup const float4 *grid')
    for index in (1,2):helper=helper.replace(f'qw3_iq3s_float_grid(grid[idx{index}])',f'grid[idx{index}]')
    assert helper==original_helper
    kernel=extract(shader,f'kernel void private_iq3_pair_grid_fast{variant}(')
    kernel=kernel.replace(f'private_iq3_pair_grid_fast{variant}','qw3_matvec_iq3_s_pair_fast')
    kernel=kernel.replace(f'private_iq3_pair_grid_dot{variant}','qw3_iq3s_dot32_vector')
    changed_grid=('''    threadgroup ushort grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = kgrid[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);''' if variant==1 else '    device const ushort *grid = kgrid;')
    assert kernel.count(changed_grid)==1
    assert kernel.replace(changed_grid,original_grid)==original_kernel
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'guarded_matrix_cases':192,'input_offsets':[16,32],'nonzero_weight_offset':2,
        'model_phases':3,'candidate_dispatches_per_continuation':608,
        'single_reused_ctx256_session':True,'native_vector_dot_expressions_loops_reductions_preserved':True,
        'static_threadgroup_bytes':[8192,1024,0],'dynamic_threadgroup_bytes':128,
        'additional_weight_or_session_data_buffers':0,'production_unchanged_during_gate':True,
        'scope':'private prefix128/greedy32 gate; eight-prefix/checkpoint gate required before adoption'}
(archive/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
