"""Validate the isolated public implementation, sequentially and without the private shader proxy."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

archive = Path(__file__).resolve().parent
root = archive.parents[2]
work = Path('/tmp/qw3-gqa-heads3-integration')
quality = json.loads((archive/'quality_v2/manifest.json').read_text())
assert quality['exit_code'] == 0 and quality['source_unchanged'] and quality['binary_unchanged']
assert 'PASS gqa_groups_quality eight_prefixes=exact checkpoint_roundtrips=16' in (archive/'quality_v2/quality.stdout').read_text()
out = archive/'public_integration_v2'
out.mkdir(exist_ok=False)
env = {k:v for k,v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
env.update(quality['environment'])
env['QW3_DENSE_MODEL'] = quality['command'][3]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest = {'started_utc':datetime.now(timezone.utc).isoformat(), 'worktree':str(work),
            'base_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip(),
            'environment':{k:v for k,v in env.items() if k.startswith(('QW3_', 'MTL_'))},
            'tests':[], 'builds':[], 'scope':'actual public selector/pipeline, matrices, eight prefixes/checkpoints and four specific agent regressions',
            'model':env['QW3_DENSE_MODEL'], 'additional_weight_and_session_data_buffers':0}
model = Path(manifest['model'])
manifest.update(model_size=model.stat().st_size, model_mtime_ns=model.stat().st_mtime_ns)

def save():
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def execute(name, command, build=False, timeout=600):
    print('starting',name,flush=True)
    start = time.monotonic()
    with (out/(name+'.stdout')).open('w') as stdout, (out/(name+'.stderr')).open('w') as stderr:
        result = subprocess.run(command,cwd=work,env=env,stdout=stdout,stderr=stderr,timeout=timeout)
    manifest['builds' if build else 'tests'].append({'name':name,'command':command,
        'exit_code':result.returncode,'seconds':time.monotonic()-start})
    save()
    assert result.returncode == 0, name
    print('passed',name,flush=True)

execute('build_public', ['make','qw3-cli','qw3-agent','qw3-bench'], True)
for suffix in ('', '_quality'):
    execute('build_matrix' if not suffix else 'build_quality',
        ['cc','-O3','-ffast-math','-mcpu=native','-Wall','-Wextra','-fobjc-arc',
         '-fno-finite-math-only','-I.','-o','/tmp/qw3-public-gqa-heads3'+suffix,
         'tests/test_public_gqa_heads3'+suffix+'.m','-lm','-pthread',
         '-framework','Foundation','-framework','Metal'], True)
assert all(not (out/(name+'.stderr')).read_text() for name in ('build_public','build_matrix','build_quality'))

names = set(json.loads((root/'docs/measurements/q4_uint_integration_2026-10-13/manifest.json').read_text())['source_sha256'])
names = {n for n in names if not n.startswith(('tests/.q4_words/', 'tests/test_q4_words_quality'))}
names.update(['tests/test_public_gqa_heads3.m','tests/test_public_gqa_heads3_quality.m',
              'tests/gqa_heads3_public/hooks.inc','tests/gqa_heads3_public/model.inc',
              'qw3_agent_unicode.h','linenoise.c','linenoise.h'])
names.update(str(p.relative_to(work)) for p in (work/'metal').glob('*.metal'))
manifest['source_sha256'] = {n:sha(work/n) for n in sorted(names)}
manifest['binary_sha256'] = {str(p):sha(p) for p in [work/'qw3-cli',work/'qw3-agent',work/'qw3-bench',
    Path('/tmp/qw3-public-gqa-heads3'),Path('/tmp/qw3-public-gqa-heads3_quality')]}
for n in sorted(names):
    target = out/'source_snapshot'/n
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(work/n,target)
(out/'integration.patch').write_text(subprocess.check_output(['git','diff','--','qw3_metal.m','metal/qw3_core_sequence.metal'],cwd=work,text=True))
save()
execute('matrix', ['caffeinate','-disu','/tmp/qw3-public-gqa-heads3','2'])
execute('quality', ['caffeinate','-disu','/tmp/qw3-public-gqa-heads3_quality',str(model),'2'])
for name, command in [('agent_bash',['sh','tests/test_agent_bash_tool.sh']),
                      ('agent_edit',['sh','tests/test_agent_edit_tool.sh']),
                      ('runtime_fingerprint',['python3','tests/test_runtime_fingerprint.py']),
                      ('dense_agent',['sh','tests/test_dense_agent.sh'])]:
    execute(name,['caffeinate','-disu',*command])
manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),
    source_unchanged=all(sha(work/n)==h for n,h in manifest['source_sha256'].items()),
    binary_unchanged=all(sha(n)==h for n,h in manifest['binary_sha256'].items()),
    model_unchanged=(model.stat().st_size,model.stat().st_mtime_ns)==(manifest['model_size'],manifest['model_mtime_ns']))
save()
assert all(manifest[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
assert 'PASS public_gqa_heads3=2 matrix_cases=432 finite_bit_exact=all' in (out/'matrix.stdout').read_text()
assert 'PASS public_gqa_heads3_quality eight_prefixes=exact checkpoint_roundtrips=16 sessions=1 additional_weight_buffer_bytes=0' in (out/'quality.stdout').read_text()
print('public integration gates passed',flush=True)
