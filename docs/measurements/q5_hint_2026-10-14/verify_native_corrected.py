"""Verify Q5 descriptor timing, corrected counters and retained partial runs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--partial',action='store_true')
parser.add_argument('--index',action='store_true')
args=parser.parse_args()
archive=Path(__file__).resolve().parent
root=archive.parents[2]
base=archive/'native_corrected'
m=json.loads((base/'manifest.json').read_text())
initial=json.loads((archive/'manifest.json').read_text())
partial=json.loads((archive/'native/manifest.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
assert m['initial_partial_native_manifest_sha256']==sha(archive/'native/manifest.json')
assert len(partial['runs'])==2 and all(r['exit_code']==0 and len(r['samples'])==3 for r in partial['runs'])
assert partial['runs'][1]['candidate_calls']==2526
assert 'AssertionError' in (archive/'native_controller.stderr').read_text()
assert m['order']==[0,1,0,2,0]
assert all(m['source_sha256'][n]==h for n,h in initial['source_sha256'].items())
assert len(m['source_sha256'])==len(initial['source_sha256'])+1
for name,h in m['source_sha256'].items():
    frozen=(base/m['source_snapshot_paths'][name]).resolve()
    assert sha(root/name)==sha(frozen)==h,name
    if args.index:
        content=subprocess.run(['git','show',':'+str(frozen.relative_to(root))],cwd=root,capture_output=True,check=True).stdout
        assert hashlib.sha256(content).hexdigest()==h,name
for name,h in {**m['binary_sha256'],**m['production_binary_sha256']}.items():assert sha(root/name)==h,name
model=Path(m['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
for location,runs in ((archive/'native',partial['runs']),(base,m['runs'])):
    for index,run in enumerate(runs):
        assert run['exit_code']==0 and run['variant']==m['order'][index]
        log=(location/(run['name']+'.stderr')).read_text()
        samples=[{'repetition':int(n),'tps':float(t),'ms':float(ms)} for n,t,ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms',log)]
        assert samples==run['samples'] and len(samples)==3
        assert [s['repetition'] for s in samples]==[1,2,3]
        assert run['candidate_calls']==(2526 if run['variant'] else 0)
        assert f"PASS q5_hint_native variant={run['variant']} calls={run['candidate_calls']} additional_weight_buffer_bytes=0" in log
        assert 'MTL_DEBUG_LAYER' not in run['environment']
        assert run['environment']['QW3_PRIVATE_Q5_HINT']==str(run['variant'])
        row=list(csv.DictReader(io.StringIO((location/(run['name']+'.stdout')).read_text())))
        assert len(row)==1
        assert all(int(row[0][k])==v for k,v in {'n_prompt':0,'n_gen':32,'n_depth':128,'reps':3,'session_bytes':176761680}.items())
        assert abs(statistics.mean(s['ms'] for s in samples)-float(row[0]['avg_ms']))<=.006
result={'index_verified':args.index,'complete':bool(m.get('completed_utc')),
        'source_references_verified':len(m['source_sha256']),
        'corrected_runs_verified':len(m['runs']),'corrected_samples_verified':3*len(m['runs']),
        'retained_initial_partial_runs':2,'retained_initial_partial_samples':6,
        'counter_expected':{'decode':97*26,'untimed_prefill_heads':4,'total':2526},
        'production_binaries_unchanged':True,'scope':'private screening; no quality/checkpoint or fresh llama comparison'}
if not args.partial:
    assert len(m['runs'])==5 and m['completed_utc']
    assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
    a=json.loads((base/'analysis.json').read_text());assert a['samples_retained']==15
    assert a['block_means_ms']==[statistics.mean(s['ms'] for s in r['samples']) for r in m['runs']]
    result['analysis']=a
    (base/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
