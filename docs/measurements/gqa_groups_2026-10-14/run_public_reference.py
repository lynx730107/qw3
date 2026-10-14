"""Fresh direct llama comparison, only after all public integration gates pass."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

b=Path(__file__).resolve().parent
root=b.parents[2]
subprocess.run(['python3',str(b/'verify_public_integration.py')],check=True,capture_output=True)
m=json.loads((b/'public_integration_v2/manifest.json').read_text())
work=Path(m['worktree'])
llama=Path('/Users/pasqualecapasso/Downloads/llama.cpp')
out=b/'public_reference'
out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta={'started_utc':datetime.now(timezone.utc).isoformat(),
      'public_manifest_sha256':sha(b/'public_integration_v2/manifest.json'),
      'public_source_sha256':m['source_sha256'],
      'public_binary_sha256':m['binary_sha256'],
      'llama_library_sha256':{str(p):sha(p) for p in sorted((llama/'build/bin').glob('*.dylib'))},
      'benchmark_driver_sha256':sha(work/'tools/bench_dense_reference.py'),
      'scope':'six AB/BA pairs per workload, all process means and raw samples retained; ordinary native decoder',
      'sessions_at_once':1}
meta['dyld_overrides_absent'] = not any(k.startswith('DYLD_') for k in os.environ)
assert meta['dyld_overrides_absent']
meta['source_snapshot_paths'] = {n:os.path.relpath(b/'public_integration_v2/source_snapshot'/n,out)
                                 for n in m['source_sha256']}
extra=['tools/bench_dense_reference.py','tests/scratch_candidates.inc','tests/scratch_matmul.inc',
       'tests/gdn_prefill_dispatch.inc','tests/gdn_prefill_candidates.metal',
       'tests/bench_packed8_scratch.m','tests/bench_scratch_prefill.m']
for name in extra:
    if not (work/name).exists():continue
    target=out/'source_snapshot'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(work/name,target)
    meta['source_snapshot_paths'][name]=str(target.relative_to(out))
shutil.copy2(work/'tools/bench_dense_reference.py',out/'bench_dense_reference.py')
command=['caffeinate','-disu','python3',str(work/'tools/bench_dense_reference.py'),
    '--model',m['model'],'--llama',str(llama/'build/bin/llama-bench'),
    '--llama-checkout',str(llama),'--qw3',str(work/'qw3-bench'),
    '--jobs','tg32_d128,pp128','--pairs','6','--repetitions','3',
    '--cooldown','60','--cooldown-scope','run','--output',str(out/'runs')]
meta['command']=command
(out/'envelope.json').write_text(json.dumps(meta,indent=2)+'\n')
result=subprocess.run(command,cwd=work)
meta.update(exit_code=result.returncode,completed_utc=datetime.now(timezone.utc).isoformat(),
    public_source_unchanged=all(sha(work/n)==h for n,h in m['source_sha256'].items()),
    public_binary_unchanged=all(sha(n)==h for n,h in m['binary_sha256'].items()),
    llama_library_unchanged=all(sha(n)==h for n,h in meta['llama_library_sha256'].items()),
    benchmark_driver_unchanged=sha(work/'tools/bench_dense_reference.py')==meta['benchmark_driver_sha256'])
(out/'envelope.json').write_text(json.dumps(meta,indent=2)+'\n')
assert result.returncode==0 and all(meta[k] for k in ('public_source_unchanged','public_binary_unchanged','llama_library_unchanged','benchmark_driver_unchanged'))
print('fresh public reference campaign finished',flush=True)
