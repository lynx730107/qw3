"""Validate the fused residual screen, exact primitive cases and model oracle."""
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
    assert len(rows)==144 and out.count('input_offset=4 guards=')==out.count('input_offset=16 guards=')==72
    for scale in ('1.0','0.5','-1.0'):assert out.count('scale='+scale+' input_offset=')==48
    assert f'PASS iq3_add={variant} matrix_cases=144 finite_bit_exact=all' in out
    log=(archive/f'matrix_v{variant}.stderr').read_text()
    assert 'Metal API Validation Enabled' in log
    assert len(re.findall(r'^iq3_add_metadata .*width=32 .*static_threadgroup_bytes=1024 dynamic_threadgroup_bytes=128 threads=64$',log,re.M))==3
out=(archive/'model.stdout').read_text()
rows=re.findall(r'^PASS iq3_add_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$',out,re.M)
assert rows==[('0','0'),('1','544'),('2','544')]

source=(root/'metal/qw3_core_moe.metal').read_text()
name='qw3_matvec_iq3_s_fast'
a=source.index('kernel void '+name+'(')
original=source[a:source.index('\n}\n',a)+3]
shader=(root/'tests/.iq3_add/kernels.metal').read_text()
for variant in (1,2):
    private=f'private_iq3_add_fast{variant}'
    a=shader.index('kernel void '+private+'(')
    body=shader[a:shader.index('\n}\n',a)+3].replace(private,name)
    body=body.replace('                                  constant float &scale,\n','')
    for i in range(4):
        expression=f'out[row] = out[row] + sum{i} * scale;' if variant==1 else f'out[row] = fma(sum{i}, scale, out[row]);'
        assert body.count(expression)==1
        body=body.replace(expression,f'out[row] = sum{i};')
    assert body==original
core=(root/'qw3.c').read_text()
private_core=(root/'tests/.iq3_add/core.inc').read_text()
insertion='''        } else if (ok && iq3_add_mode &&
                   lw->ffn_down->type == QW3_TENSOR_IQ3_S &&
                   n_ff == 17408u && n_embd == 5120u) {
            ok = private_iq3_inner_add_x0(
                s->metal, lw->ffn_down->offset, n_ff, n_embd);
'''
assert private_core.count(insertion)==1 and private_core.replace(insertion,'')==core
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'guarded_matrix_cases':288,'input_offsets':[4,16],'scales':[1.0,0.5,-1.0],
        'model_phases':3,'candidate_dispatches_per_continuation':544,
        'single_reused_ctx256_session':True,'native_dot_helper_loop_and_reduction_preserved':True,
        'private_core_has_only_targeted_ffn_branch':True,'production_unchanged_during_gate':True,
        'scope':'private prefix128/greedy32 gate; eight-prefix/checkpoint gate required before adoption'}
(archive/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
