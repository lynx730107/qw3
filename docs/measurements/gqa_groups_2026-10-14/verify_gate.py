"""Verify generic one-head failure and the complete three-head numerical gate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', action='store_true')
args = parser.parse_args()
b = Path(__file__).resolve().parent
root = b.parents[2]
parent = json.loads((b / 'manifest.json').read_text())
selected = json.loads((b / 'selected_v2_corrected/manifest.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


for directory, m in ((b,parent), (b/'selected_v2_corrected',selected)):
    assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged','store_unchanged'))
    for name, digest in m['source_sha256'].items():
        frozen = (directory / m['source_snapshot_paths'][name]).resolve()
        assert sha(root / name) == sha(frozen) == digest, name
        if args.index:
            data = subprocess.run(['git','show',':'+str(frozen.relative_to(root))],cwd=root,capture_output=True,check=True).stdout
            assert hashlib.sha256(data).hexdigest() == digest
    for name, digest in m['production_binary_sha256'].items():
        assert sha(root / name) == digest
assert not parent['all_passed'] and len(parent['runs']) == 2
assert [(r['name'],r['exit_code'],r['matrix_cases_passed']) for r in parent['runs']] == [('matrix_v1',1,6),('matrix_v2',0,432)]
log = (b / 'matrix_v1.stderr').read_text()
failure = re.search(r'bits=(38da8c97)/(38da8cb0)',log)
assert failure and abs(int(failure[1],16)-int(failure[2],16)) == 25
out = (b / 'matrix_v2.stdout').read_text()
assert len(re.findall(r'^matrix_exact variant=2 .*guards=exact input=unchanged$',out,re.M)) == 432
assert 'PASS gqa_groups=2 matrix_cases=432 finite_bit_exact=all' in out
assert 'gqa_groups_dispatch_counts calls=432 fallbacks=0' in (b/'matrix_v2.stderr').read_text()
assert not (b/'build.stderr').read_text() and not (b/'model2_build.stderr').read_text()
assert selected['all_passed'] and selected['selected_variant']==2 and selected['exit_code']==0
assert sha(Path(selected['binary'])) == selected['binary_sha256']
out = (b/'selected_v2_corrected/model.stdout').read_text()
assert 'selected_gqa_variant=2' in out
assert re.findall(r'^PASS gqa_groups_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$',out,re.M) == [('0','0'),('1','512')]
assert 'Metal API Validation Enabled' in (b/'selected_v2_corrected/model.stderr').read_text()
assert 'phase<=1' in (root/'tests/.gqa_groups/model2.inc').read_text()
assert 'groups_variant=phase ? 2 : 1' in (root/'tests/.gqa_groups/model2.inc').read_text()
native = (root/'metal/qw3_core_sequence.metal').read_text()
a = native.index('kernel void qw3_gqa_attend_n_inner(')
original = native[a:native.index('\n}\n',a)+3]
shader = (root/'tests/.gqa_groups/kernels.metal').read_text()
for v,cap in ((1,1),(2,3)):
    name=f'private_gqa_groups{v}'
    a=shader.index('kernel void '+name+'(')
    body=shader[a:shader.index('\n}\n',a)+3]
    prelude=f'''    if (args.n_kv_heads == 0u) return;
    const uint full_heads = args.n_heads / args.n_kv_heads;
    if (full_heads == 0u || full_heads > 8u) return;
    const uint chunks = (full_heads + {cap-1}u) / {cap}u;
    const uint kvh = h / chunks;
    const uint head_chunk = h % chunks;
'''
    body=body.replace(name,'qw3_gqa_attend_n_inner')
    body=body.replace(prelude+'    if (kvh >= args.n_kv_heads','    if (h >= args.n_kv_heads')
    body=body.replace('    uint i = uint(tid);\n','    uint i = uint(tid);\n    uint kvh = h;\n')
    body=body.replace(f'    uint group_heads = min({cap}u, full_heads - head_chunk * {cap}u);','    uint group_heads = args.n_heads / args.n_kv_heads;')
    body=body.replace(f'    uint first_qh = kvh * full_heads + head_chunk * {cap}u;','    uint first_qh = kvh * group_heads;')
    body=body.replace(f'gh < {cap}u','gh < 8u')
    for array in ('qv','max_score','denom','acc'):
        body=body.replace(f'float {array}[{cap}];',f'float {array}[8];')
    assert body == original
result={'index_verified':args.index,'selected_variant':2,'selected_matrix_cases':432,
        'rejected_variant':1,'first_failure_ulp_distance':25,'rejected_completed_cases':6,
        'selected_model_phases':2,'selected_model_dispatches':512,'single_reused_ctx256_session':True,
        'dynamic_threadgroup_bytes_before_after':[256,256],'static_threadgroup_bytes_before_after':[0,0],
        'new_weight_or_session_data_buffers':0,'model_threadgroups_before_after':[4,8],
        'per_head_expressions_reductions_temporal_order_preserved':True,
        'scope':'three-head candidate passes available numerical gate; extended prefixes/checkpoints and performance remain unproven'}
(b/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
