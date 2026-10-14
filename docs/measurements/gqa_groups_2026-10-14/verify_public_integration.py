"""Recheck actual public shader selection and the integrated numerical/agent gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index',action='store_true')
args=parser.parse_args()
b=Path(__file__).resolve().parent
root=b.parents[2]
base=b/'public_integration_v2'
m=json.loads((base/'manifest.json').read_text())
work=Path(m['worktree'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert m['completed_utc'] and all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
assert len(m['builds'])==3 and len(m['tests'])==6
assert all(t['exit_code']==0 for t in m['builds']+m['tests'])
assert [t['name'] for t in m['tests']]==['matrix','quality','agent_bash','agent_edit','runtime_fingerprint','dense_agent']
for n,h in m['source_sha256'].items():
    frozen=base/'source_snapshot'/n
    assert sha(frozen)==sha(work/n)==h,n
    if args.index:
        data=subprocess.check_output(['git','show',':'+str(frozen.relative_to(root))],cwd=root)
        assert hashlib.sha256(data).hexdigest()==h,n
for n,h in m['binary_sha256'].items():assert sha(n)==h,n
model=Path(m['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
private=(root/'tests/.gqa_groups/kernels.metal').read_text()
start=private.index('kernel void private_gqa_groups2(')
candidate=private[start:].strip().replace('private_gqa_groups2','qw3_gqa_attend_n_inner_heads3')
public=(work/'metal/qw3_core_sequence.metal').read_text()
assert public[public.index('kernel void qw3_gqa_attend_n_inner_heads3('):].strip()==candidate
matrix=(base/'matrix.stdout').read_text()
assert len(re.findall(r'^matrix_exact ',matrix,re.M))==432
assert 'PASS public_gqa_heads3=2 matrix_cases=432 finite_bit_exact=all' in matrix
assert 'public_heads3_dispatch_counts calls=18 fallbacks=0' in (base/'matrix.stderr').read_text()
quality=(base/'quality.stdout').read_text()
rows=re.findall(r'full_exact phase=(\d) prefix=(\d+) extended_prefix=(\d+) continuation=32 finite_logits=bit_exact canonical_state=exact history=exact checkpoint=exact calls=(\d+) sessions=1',quality)
assert rows==[(str(phase),str(length),str(159 if length==128 else length),str(512 if phase else 0))
    for length in (1,15,16,31,127,128,129,159) for phase in (0,1)]
assert 'PASS public_gqa_heads3_quality eight_prefixes=exact checkpoint_roundtrips=16 sessions=1 additional_weight_buffer_bytes=0' in quality
for name,marker in [('agent_bash','test-agent-bash: ok'),('agent_edit','test-agent-edit: ok'),
                    ('dense_agent','test-dense-agent: ok')]:
    assert marker in (base/(name+'.stdout')).read_text(),name
result={'passed':True,'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'fixture_cases':432,'actual_public_pipeline_selections_fixture':18,
        'prefix_phases':16,'checkpoint_roundtrips':16,'specific_agent_regressions':4,
        'scope':m['scope'],'total_peak_ram_measured':False}
(base/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
