"""After the live reference campaign, test cap-two/cap-one against public cap-three.

Each candidate must pass its eligible matrices before loading the model.
This is a numerical screen, not an extended checkpoint or performance gate.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

b=Path(__file__).resolve().parent
root=b.parents[2]
previous=b.parent/'gqa_groups_2026-10-14'
reference=json.loads((previous/'public_reference/envelope.json').read_text())
assert reference.get('completed_utc') and reference['exit_code']==0
subprocess.run(['python3',str(previous/'verify_public_reference.py')],check=True,capture_output=True)
public=json.loads((previous/'public_integration_v2/manifest.json').read_text())
out=b/'numerical_v2'
out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
env.update(QW3_METAL_KV_F16='1',QW3_METAL_KV_Q8_0='0',QW3_METAL_NGL='64',
    QW3_METAL_DENSE_PREFILL_FAST='1',QW3_METAL_DENSE_PREFILL_K64='1',
    QW3_METAL_DENSE_GDN_CACHE='1',MTL_DEBUG_LAYER='1')
model=Path(public['model'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'started_utc':datetime.now(timezone.utc).isoformat(),
   'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
   'environment':{k:v for k,v in env.items() if k.startswith(('QW3_','MTL_'))},
   'variants':{'0':{'capacity':3,'groups':8,'scope':'current public baseline'},
               '1':{'capacity':2,'groups':12,'scope':'private eligible 27B F16 route only'},
               '2':{'capacity':1,'groups':24,'scope':'private eligible 27B F16 route only; no generic claim'}},
   'model':str(model),'model_size':model.stat().st_size,'model_mtime_ns':model.stat().st_mtime_ns,
   'sessions_at_once':1,'additional_weight_and_session_data_buffers':0,
   'acceptance_policy_version':2,'agent_behavior_gate_passed':False,
   'adoption_permitted':False,
   'production_binary_sha256':{n:sha(root/n) for n in ('qw3-cli','qw3-agent','qw3-bench')},
   'builds':[],'tests':[]}
def save(): (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def execute(name,command,timeout=300):
    print('starting',name,flush=True)
    start=time.monotonic()
    with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
        p=subprocess.run(command,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=timeout)
    return {'name':name,'command':command,'exit_code':p.returncode,'seconds':time.monotonic()-start}

for name,command in [('generate',['python3','tests/prepare_gqa_heads_probe.py']),
                     ('fingerprint',['make','qw3_runtime_fingerprint.h'])]:
    result=execute(name,command);m['builds'].append(result);save();assert result['exit_code']==0
names=set(public['source_sha256'])
names.update(str(p.relative_to(root)) for p in (root/'tests/.gqa_heads_probe').iterdir())
names.update(['tests/prepare_gqa_heads_probe.py','tests/test_gqa_heads_probe.m',
    'tests/test_gqa_heads_probe_model.m','tests/test_gqa_heads_probe_quality.m','tests/bench_gqa_heads_probe_native.m',
    'tests/optimization_acceptance.h','tests/gqa_heads_numeric_oracles.inc',
    'tests/test_optimization_acceptance.c',
    'docs/measurements/gqa_heads_probe_2026-10-14/run_numerical.py',
    'docs/measurements/gqa_groups_2026-10-14/source_snapshot/tests/.gqa_groups/native.inc'])
m['source_sha256']={n:sha(root/n) for n in sorted(names)}
m['source_snapshot_paths']={}
for n,h in m['source_sha256'].items():
    if public['source_sha256'].get(n)==h:
        target=previous/'public_integration_v2/source_snapshot'/n
    else:
        target=out/'source_snapshot'/n
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/n,target)
    m['source_snapshot_paths'][n]=os.path.relpath(target,out)
save()
result=execute('build_acceptance',['cc','-O2','-Wall','-Wextra','-fno-finite-math-only',
    '-o','/tmp/qw3-optimization-acceptance-v2-test','tests/test_optimization_acceptance.c','-lm'])
m['builds'].append(result);save();assert result['exit_code']==0
result=execute('acceptance_contract',['/tmp/qw3-optimization-acceptance-v2-test'])
m['tests'].append(result);save();assert result['exit_code']==0
binaries={}
for label,source in [('matrix','tests/test_gqa_heads_probe.m'),
                     ('model','tests/test_gqa_heads_probe_model.m'),
                     ('native','tests/bench_gqa_heads_probe_native.m')]:
    binary=Path('/tmp/qw3-gqa-heads-probe-v2-'+label)
    command=['cc','-O3','-ffast-math','-mcpu=native','-Wall','-Wextra','-fobjc-arc',
        '-fno-finite-math-only','-I.','-o',str(binary),source,'-lm','-pthread',
        '-framework','Foundation','-framework','Metal']
    result=execute('build_'+label,command);m['builds'].append(result);save()
    assert result['exit_code']==0 and not (out/('build_'+label+'.stderr')).read_text()
    binaries[label]=str(binary)
m['binary_sha256']={n:sha(n) for n in binaries.values()};save()
for variant in (1,2):
    result=execute(f'matrix_v{variant}',['caffeinate','-disu',binaries['matrix'],str(variant)])
    m['tests'].append(result);save()
    if result['exit_code']:
        result['model_skipped']='matrix failure; no timing permitted';save();continue
    text=(out/(result['name']+'.stdout')).read_text()
    assert f'PASS gqa_heads_probe={variant} matrix_cases=432 finite_numerically_valid=all policy=2' in text
    assert 'gqa_heads_probe_dispatch_counts calls=18 fallbacks=0' in (out/(result['name']+'.stderr')).read_text()
    result=execute(f'model_v{variant}',['caffeinate','-disu',binaries['model'],str(model),str(variant)])
    m['tests'].append(result);save()
    if result['exit_code']:continue
    rows=re.findall(r'PASS gqa_heads_probe_model phase=(\d).*calls=(\d+)',(out/(result['name']+'.stdout')).read_text())
    assert rows==[('0','0'),('1','512')]
    assert 'numeric_policy=2 ' in (out/(result['name']+'.stdout')).read_text()
    assert 'gqa_heads_probe_dispatch_counts calls=512 fallbacks=0' in (out/(result['name']+'.stderr')).read_text()
    print('numerical screen passed variant',variant,flush=True)
m.update(completed_utc=datetime.now(timezone.utc).isoformat(),
    source_unchanged=all(sha(root/n)==h for n,h in m['source_sha256'].items()),
    binary_unchanged=all(sha(n)==h for n,h in m['binary_sha256'].items()),
    production_binary_unchanged=all(sha(root/n)==h for n,h in m['production_binary_sha256'].items()),
    model_unchanged=(model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns']))
save()
assert all(m[k] for k in ('source_unchanged','binary_unchanged','production_binary_unchanged','model_unchanged'))
print('numerical screen completed; inspect per-variant outcomes before timing',flush=True)
