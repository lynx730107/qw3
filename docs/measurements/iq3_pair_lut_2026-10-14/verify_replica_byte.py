"""Verify raw balanced process runs and recompute the paired uncertainty."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index',action='store_true')
parser.add_argument('--partial',action='store_true')
args=parser.parse_args()
archive=Path(__file__).resolve().parent
root=archive.parents[2]
base=archive/'replica_byte'
m=json.loads((base/'manifest.json').read_text())
initial=json.loads((archive/'manifest.json').read_text())
order=[0,2,2,0,0,2,2,0,0,2,2,0]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
assert m['order']==order and m['cooldown_seconds_by_process']==[60]*12
assert m['context']==256 and m['gpu_layers']==64 and m['kv']=='f16'
assert m['repetitions']==3 and m['warmup']
assert m['initial_native_manifest_sha256']==sha(archive/'native/manifest.json')
assert m['initial_native_analysis_sha256']==sha(archive/'native/analysis.json')
assert m['previous_half_manifest_sha256']==sha(archive/'replica_half/manifest.json')
assert m['previous_half_analysis_sha256']==sha(archive/'replica_half/analysis.json')
assert all(m['source_sha256'][name]==value for name,value in initial['source_sha256'].items())
assert len(m['source_sha256'])==len(initial['source_sha256'])+1
for name,expected in m['source_sha256'].items():
    frozen=(base/m['source_snapshot_paths'][name]).resolve()
    assert sha(root/name)==sha(frozen)==expected,name
    if args.index:
        content=subprocess.run(['git','show',':'+str(frozen.relative_to(root))],cwd=root,capture_output=True,check=True).stdout
        assert hashlib.sha256(content).hexdigest()==expected,name
for name,value in {**m['binary_sha256'],**m['production_binary_sha256']}.items():assert sha(root/name)==value,name
model=Path(m['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
means=[]
for index,run in enumerate(m['runs']):
    assert run['exit_code']==0 and run['variant']==order[index]
    assert run['name']==f"block_{index}_v{run['variant']}"
    assert 'MTL_DEBUG_LAYER' not in run['environment']
    assert run['environment']['QW3_PRIVATE_IQ3_PAIR_LUT']==str(run['variant'])
    expected=['caffeinate','-disu','./qw3-iq3-pair-lut-native-bench','-m',str(model),'--metal','--ngl','64',
              '--llama-style','--ctx-alloc','256','-p','0','-n','32','-d','128','-r','3','--seed','1','-t','4']
    assert run['command']==expected
    log=(base/(run['name']+'.stderr')).read_text()
    samples=[{'repetition':int(n),'tps':float(t),'ms':float(ms)} for n,t,ms in
             re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms',log)]
    assert samples==run['samples'] and len(samples)==3
    assert [s['repetition'] for s in samples]==[1,2,3]
    assert all(math.isfinite(s[k]) and s[k]>0 for s in samples for k in ('tps','ms'))
    assert run['candidate_calls']==(1843 if run['variant'] else 0)
    assert f"PASS iq3_pair_lut_native variant={run['variant']} calls={run['candidate_calls']} additional_weight_buffer_bytes=0" in log
    rows=list(csv.DictReader(io.StringIO((base/(run['name']+'.stdout')).read_text())))
    assert len(rows)==1
    assert all(int(rows[0][k])==v for k,v in {'n_prompt':0,'n_gen':32,'n_depth':128,'reps':3,'session_bytes':176761680}.items())
    mean=statistics.mean(s['ms'] for s in samples)
    assert abs(mean-float(rows[0]['avg_ms']))<=.006
    means.append(mean)
result={'index_verified':args.index,'complete':bool(m.get('completed_utc')),
        'source_references_verified':len(m['source_sha256']),'validated_runs':len(means),
        'validated_samples':3*len(means),'sources_snapshots_binaries_model_match':True,
        'scope':'six paired process-run means, not the 36 individual repetitions; no fresh llama comparison'}
if not args.partial:
    assert len(means)==12 and m['completed_utc']
    assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
    a=json.loads((base/'analysis.json').read_text())
    saved=[];off=[];on=[]
    for index in range(6):
        values={order[2*index+j]:means[2*index+j] for j in (0,1)}
        off.append(values[0]);on.append(values[2]);saved.append(values[0]-values[2])
    mean=statistics.mean(saved);radius=2.571*statistics.stdev(saved)/math.sqrt(6)
    expected={'saved_ms_mean':mean,'candidate_mean_ms':statistics.mean(on),'control_mean_ms':statistics.mean(off),
              'throughput_change_percent':100*(statistics.mean(off)/statistics.mean(on)-1),
              'off_spread_percent':100*(max(off)-min(off))/statistics.mean(off),
              'on_spread_percent':100*(max(on)-min(on))/statistics.mean(on)}
    for key,value in expected.items():assert math.isclose(a[key],value,rel_tol=1e-12,abs_tol=1e-9),key
    assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-9) for x,y in zip(a['saved_ms_ci95'],[mean-radius,mean+radius]))
    assert a['samples_retained']==36 and a['run_means_ms']==means
    assert a['candidate_wins_all_pairs']==all(value>0 for value in saved)
    groups={str(first):statistics.mean(saved[i] for i in range(6) if order[2*i]==first) for first in (0,2)}
    assert a['order_group_saved_ms']==groups
    assert a['both_order_groups_positive']==all(value>0 for value in groups.values())
    assert a['ci_lower_positive']==(mean-radius>0)
    stable=a['off_spread_percent']<=5 and a['on_spread_percent']<=5
    result.update(analysis=a,performance_gate_passed=stable and a['ci_lower_positive'] and a['both_order_groups_positive'],
                  swap_before_after_each_run_equal=all(r['swap_before']==r['swap_after'] for r in m['runs']),
                  swap_usage_values=sorted({r[k] for r in m['runs'] for k in ('swap_before','swap_after')}))
    (base/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
