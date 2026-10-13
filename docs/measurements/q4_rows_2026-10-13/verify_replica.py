"""Verify the independent replica against immutable inputs and raw logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
p=argparse.ArgumentParser();p.add_argument('--index',action='store_true');args=p.parse_args()
b=Path(__file__).resolve().parent;root=b.parents[2];rbase=b/'replica'
m=json.loads((rbase/'manifest.json').read_text());initial=json.loads((b/'manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert m['order']==[0,1,1,0,0,1,1,0,0,1,1,0] and len(m['runs'])==12
assert m['cooldown_seconds_by_process']==[180]+[60]*11
assert m['initial_native_manifest_sha256']==sha(b/'native/manifest.json')
assert m['initial_native_analysis_sha256']==sha(b/'native/analysis.json')
assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
assert all(m['source_sha256'][n]==h for n,h in initial['source_sha256'].items())
assert len(m['source_sha256'])==len(initial['source_sha256'])+1
for name,h in m['source_sha256'].items():
    frozen=(rbase/m['source_snapshot_paths'][name]).resolve()
    assert sha(frozen)==h and sha(root/name)==h,name
    if args.index:
        data=subprocess.check_output(['git','show',':'+str(frozen.relative_to(root))],cwd=root)
        assert hashlib.sha256(data).hexdigest()==h,('index',name)
for name,h in {**m['binary_sha256'],**m['production_binary_sha256']}.items():assert sha(root/name)==h,name
model=Path(m['model']);assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
for i,r in enumerate(m['runs']):
    assert r['exit_code']==0 and r['variant']==m['order'][i] and len(r['samples'])==3
    assert r['environment'].get('MTL_DEBUG_LAYER') is None
    log=(rbase/(r['name']+'.stderr')).read_text()
    samples=[{'repetition':int(n),'tps':float(t),'ms':float(ms)} for n,t,ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms',log)]
    assert samples==r['samples']
    counts=re.search(r'q4_rows_dispatch_counts calls=(\d+) fallbacks=(\d+)',log)
    assert counts and tuple(map(int,counts.groups()))==((3298,194) if r['variant'] else (0,0))
    assert r['candidate_calls']==(3298 if r['variant'] else 0)
    rows=[x for x in (rbase/(r['name']+'.stdout')).read_text().splitlines() if x.startswith('tg,')]
    assert len(rows)==1 and int(rows[0].split(',')[-1])==176761680
analysis=json.loads((rbase/'analysis.json').read_text());assert analysis['samples_retained']==36
result={'index_verified':args.index,'frozen_references_verified':len(m['source_sha256']),
        'ordinary_samples_verified':36,'process_pairs':6,'production_unchanged':True,
        'scope':'private replica; no quality/checkpoint runtime gate or fresh llama comparison',
        'analysis':analysis}
(rbase/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
