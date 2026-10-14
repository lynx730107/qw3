"""Run the one-session checkpoint gate only after a positive balanced replica."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
b=Path(__file__).resolve().parent;root=b.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--performance-analysis',default=None)
parser.add_argument('--variant',type=int,choices=(1,2),default=2)
args=parser.parse_args()
assert args.variant==2
m_dir=b/'selected_v2_corrected'
m=json.loads((m_dir/'manifest.json').read_text())
performance_path=(b/(args.performance_analysis or f'replica_v{args.variant}/analysis.json')).resolve()
a=json.loads(performance_path.read_text())
replica=json.loads((performance_path.parent/'manifest.json').read_text())
assert replica['completed_utc'] and len(replica['runs'])==12
assert all(r['exit_code']==0 for r in replica['runs'])
assert {r['variant'] for r in replica['runs'] if r['variant']}=={args.variant}
assert all(replica[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
assert a['ci_lower_positive'] and a['both_order_groups_positive']
assert a['off_spread_percent']<=5 and a['on_spread_percent']<=5
archive=b/f'quality_v{args.variant}';archive.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((b/'quality_build.json').read_text())
assert meta['variant']==args.variant
binary=Path('/tmp/qw3-gqa-groups-quality-test')
assert sha(binary)==meta['binary_sha256']
assert sha(root/'tests/test_gqa_groups_quality.m')==meta['source_sha256']
paths=list(m['source_sha256'])+['tests/test_gqa_groups_quality.m','docs/measurements/gqa_groups_2026-10-14/run_quality.py']
sources={n:sha(root/n) for n in paths};snapshots={}
for n,h in sources.items():
    if m['source_sha256'].get(n)==h:
        frozen=(m_dir/m['source_snapshot_paths'][n]).resolve()
        snapshots[n]=os.path.relpath(frozen,archive)
    else:
        frozen=archive/'source_snapshot'/n;frozen.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/n,frozen);snapshots[n]=str(frozen.relative_to(archive))
env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
env.update(m['environment'])
command=['caffeinate','-disu',str(binary),m['model'],str(args.variant)]
result={'started_utc':datetime.now(timezone.utc).isoformat(),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'variant':args.variant,'performance_analysis':str(performance_path.relative_to(b)),
        'performance_analysis_sha256':sha(performance_path),
        'source_sha256':sources,'source_snapshot_paths':snapshots,'command':command,'environment':m['environment'],
        'binary_sha256':meta['binary_sha256'],'model_size':m['model_size'],'model_mtime_ns':m['model_mtime_ns'],
        'context':256,'sessions':1,'additional_weight_and_session_data_buffers':0,
        'scope':'eight prefixes and sixteen checkpoint roundtrips, one reused session and disk oracles; not agent regressions'}
result['swap_before']=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip()
start=time.monotonic()
with (archive/'quality.stdout').open('w') as o,(archive/'quality.stderr').open('w') as e:
    process=subprocess.run(command,cwd=root,env=env,stdout=o,stderr=e,timeout=300)
model=Path(m['model'])
result.update(exit_code=process.returncode,seconds=time.monotonic()-start,
              model_unchanged=(model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns']),
              source_unchanged=all(sha(root/n)==h for n,h in sources.items()),
              binary_unchanged=sha(binary)==meta['binary_sha256'],
              swap_after=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip())
(archive/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
assert process.returncode==0 and result['source_unchanged'] and result['binary_unchanged'] and result['model_unchanged']
assert 'PASS gqa_groups_quality eight_prefixes=exact checkpoint_roundtrips=16 sessions=1 additional_weight_buffer_bytes=0' in (archive/'quality.stdout').read_text()
print('eight-prefix/checkpoint gate passed',flush=True)
