"""Independent matrix gates, then one-session disk-oracle checks for survivors."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

archive = Path(__file__).resolve().parent
root = archive.parents[2]
previous = archive.parent / 'iq3_immutable_2026-10-11'
old = json.loads((previous / 'manifest.json').read_text())
model = Path(old['model'])
assert not (archive / 'build.stderr').read_text()
assert (model.stat().st_size,model.stat().st_mtime_ns)==(old['model_size'],old['model_mtime_ns'])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def swap():
    return subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip()

paths = list(old['source_sha256']) + [
    'tests/prepare_gdn_two128.py','tests/gdn_two128_candidates.inc','tests/test_gdn_two128.m',
    'tests/test_gdn_two128_model.m','tests/bench_gdn_two128_native.m',
    *('tests/.gdn_two128/'+s for s in ('kernels.metal','model.inc','native.inc'))]
sources = {p:sha(root/p) for p in paths}
snapshots = {}
for name,digest in sources.items():
    if old['source_sha256'].get(name)==digest:
        frozen=(previous/old['source_snapshot_paths'][name]).resolve()
        assert sha(frozen)==digest
    else:
        frozen=archive/'source_snapshot'/name
        frozen.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,frozen)
    snapshots[name]=os.path.relpath(frozen,archive)
binaries={name:sha(root/name) for name in ('qw3-gdn-two128-test','qw3-gdn-two128-model-test','qw3-gdn-two128-native-bench')}
production={name:sha(root/name) for name in old['production_binary_sha256']}
assert production==old['production_binary_sha256']
settings=old['environment']
env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
env.update(settings)
m={'started_utc':now(),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
   'scope':'two columns in 64 active lanes; native 128-thread RMS; existing inner buffer used for intermediate cores',
   'variants':{'1':'cache32','2':'cache64'},'additional_weight_buffer_bytes':0,
   'additional_session_data_buffers':0,'dynamic_threadgroup_bytes':128,'dispatch_threads':128,
   'model':str(model),'model_size':old['model_size'],'model_mtime_ns':old['model_mtime_ns'],
   'source_sha256':sources,'source_snapshot_paths':snapshots,'binary_sha256':binaries,
   'production_binary_sha256':production,'environment':settings,'runs':[],
   'matrix_passed_variants':[],'qualified_variants':[],'build_warning_free':True}

def save():
    (archive/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')

def run(name,variant,args):
    command=['caffeinate','-disu',*args]
    before=swap();start=time.monotonic()
    with (archive/(name+'.stdout')).open('w') as out,(archive/(name+'.stderr')).open('w') as err:
        result=subprocess.run(command,cwd=root,env=env,stdout=out,stderr=err,timeout=300)
    r={'name':name,'variant':variant,'command':command,'exit_code':result.returncode,
       'seconds':time.monotonic()-start,'swap_before':before,'swap_after':swap()}
    m['runs'].append(r);save()
    print('finished',name,'exit',result.returncode,flush=True)
    return result.returncode,(archive/(name+'.stdout')).read_text()

save()
try:
    for variant in (1,2):
        code,out=run(f'matrix_v{variant}',variant,['./qw3-gdn-two128-test',str(model),str(variant)])
        if not code:
            assert len(re.findall(r'^matrix_exact .*guards=ok inputs=unchanged$',out,re.M))==9
            assert f'PASS gdn_two128_matrix variant={variant} cases=9' in out
            m['matrix_passed_variants'].append(variant);save()
    for variant in m['matrix_passed_variants']:
        code,out=run(f'model_v{variant}',variant,['./qw3-gdn-two128-model-test',str(model),str(variant)])
        if not code:
            phases=re.findall(r'^PASS gdn_two128_model phase=(\d+) .*calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$',out,re.M)
            assert phases==[('0','0'),(str(variant),'1536')]
            m['qualified_variants'].append(variant);save()
finally:
    m.update(completed_utc=now(),source_unchanged=all(sha(root/p)==h for p,h in sources.items()),
             binary_unchanged=all(sha(root/p)==h for p,h in {**binaries,**production}.items()),
             model_unchanged=(model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns']))
    m['all_passed']=m['qualified_variants']==[1,2]
    save()
assert m['source_unchanged'] and m['binary_unchanged'] and m['model_unchanged']
print('qualified_variants',m['qualified_variants'],'production unchanged',flush=True)
