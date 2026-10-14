"""Recheck v2 numerical limits, actual selection, frozen sources and adoption lock."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index',action='store_true')
args=parser.parse_args()
b=Path(__file__).resolve().parent;root=b.parents[2];base=b/'numerical_v2'
m=json.loads((base/'manifest.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert m['completed_utc'] and m['acceptance_policy_version']==2
assert m['agent_behavior_gate_passed'] is False and m['adoption_permitted'] is False
assert all(m[k] for k in ('source_unchanged','binary_unchanged','production_binary_unchanged','model_unchanged'))
assert all(r['exit_code']==0 for r in m['builds']+m['tests'])
assert m['sessions_at_once']==1 and m['additional_weight_and_session_data_buffers']==0
for name,h in m['source_sha256'].items():
    frozen=(base/m['source_snapshot_paths'][name]).resolve()
    assert sha(root/name)==sha(frozen)==h,name
    if args.index:
        data=subprocess.check_output(['git','show',':'+str(frozen.relative_to(root))],cwd=root)
        assert hashlib.sha256(data).hexdigest()==h,name
for name,h in m['binary_sha256'].items():assert sha(name)==h,name
for name,h in m['production_binary_sha256'].items():assert sha(root/name)==h,name
model=Path(m['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
assert 'rounding=accepted substantive_error=rejected NaN=rejected signed_zero=accepted' in (base/'acceptance_contract.stdout').read_text()
results=[]
for v in (1,2):
    text=(base/f'matrix_v{v}.stdout').read_text()
    assert len(re.findall(r'^matrix_numeric_valid ',text,re.M))==432
    assert f'PASS gqa_heads_probe={v} matrix_cases=432 finite_numerically_valid=all policy=2' in text
    rows=re.findall(r'matrix_numeric variant=(\d+) nmse=(\S+) rmse=(\S+) max_abs=(\S+) different_bits=(\d+)',text)
    assert len(rows)==432 and all(int(r[0])==v for r in rows)
    assert all(math.isfinite(float(r[i])) and float(r[i])>=0 for r in rows for i in (1,2,3))
    assert all(float(r[1])<=1e-7 for r in rows)
    assert 'gqa_heads_probe_dispatch_counts calls=18 fallbacks=0' in (base/f'matrix_v{v}.stderr').read_text()
    model_text=(base/f'model_v{v}.stdout').read_text()
    values=re.findall(r'logits_numeric valid=(\d) rmse=(\S+) max_abs=(\S+) nmse=(\S+) different_bits=(\d+) argmax_match=(\d)',model_text)
    assert len(values)==33
    assert all(r[0]=='1' and float(r[1])<=1e-3 and float(r[2])<=5e-3 for r in values)
    assert all(math.isfinite(float(r[i])) and float(r[i])>=0 for r in values for i in (1,2,3))
    phases=re.findall(r'PASS gqa_heads_probe_model phase=(\d).*calls=(\d+)',model_text)
    assert phases==[('0','0'),('1','512')]
    assert model_text.count('input_tokens=teacher_forced history=exact')==2
    assert 'gqa_heads_probe_dispatch_counts calls=512 fallbacks=0' in (base/f'model_v{v}.stderr').read_text()
    summary=re.search(r'numeric_policy=2 logits_vectors=33 logits_rmse=(\S+) logits_max_abs=(\S+) logits_different_bits=(\d+) argmax_changes=(\d+) state_nmse=(\S+) state_different_bits=(\d+).*agent_behavior=not_tested',model_text)
    assert summary and float(summary[5])<=1e-7
    results.append({'variant':v,'head_capacity':2 if v==1 else 1,
        'matrix_cases':432,'matrix_actual_candidate_selections':18,
        'matrix_max_nmse':max(float(r[1]) for r in rows),
        'matrix_max_abs':max(float(r[3]) for r in rows),
        'matrix_different_bits':sum(int(r[4]) for r in rows),
        'logit_vectors':33,'logits_rmse':float(summary[1]),'logits_max_abs':float(summary[2]),
        'logits_different_bits':int(summary[3]),'argmax_changes':int(summary[4]),
        'state_nmse':float(summary[5]),'state_different_bits':int(summary[6]),
        'screen_passed':True,'agent_behavior_gate_passed':False,'adoption_permitted':False})
report={'passed':True,'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'variants':results,'scope':'numerical screen only, one synthetic prefix128 and 32 teacher-forced steps; not agent quality or performance',
        'checkpoint_roundtrips_tested':0,'total_peak_ram_measured':False}
(base/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
