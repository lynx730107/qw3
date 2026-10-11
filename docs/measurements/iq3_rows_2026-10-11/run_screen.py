"""Sequential numerical screening; original weights and one reused session."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[3]
assert not (Path(__file__).resolve().parent / 'build.stderr').read_text()
archive = Path(__file__).resolve().parent
previous = root / 'docs/measurements/gdn_two128_2026-10-11'
old = json.loads((previous / 'manifest.json').read_text())
model = Path(old['model'])
stat_before = (model.stat().st_size, model.stat().st_mtime_ns)
assert stat_before == (old['model_size'], old['model_mtime_ns'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def swap():
    return subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()


paths = list(old['source_sha256']) + [
    'tests/prepare_iq3_rows.py', 'tests/test_iq3_rows.m',
    'tests/test_iq3_rows_model.m', 'tests/bench_iq3_rows_native.m',
    *('tests/.iq3_rows/' + s for s in ['hooks.inc', 'core.inc', 'kernels.metal', 'matrix.inc', 'model.inc', 'native.inc']),
]
sources = {p: sha(root / p) for p in paths}
snapshots = {}
for name, digest in sources.items():
    if name in old['source_sha256'] and digest == old['source_sha256'][name]:
        frozen = (previous / old['source_snapshot_paths'][name]).resolve()
        assert sha(frozen) == digest
    else:
        frozen = archive / 'source_snapshot' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, frozen)
    snapshots[name] = os.path.relpath(frozen, archive)
binaries = {p: sha(root / p) for p in [
    'qw3-iq3-rows-test', 'qw3-iq3-rows-model-test', 'qw3-iq3-rows-native-bench']}
production = {p: sha(root / p) for p in ['qw3-bench', 'qw3-agent', 'qw3-cli']}
assert production['qw3-bench'] == old['production_binary_sha256']['qw3-bench']
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
settings = {'QW3_METAL_KV_F16': '1', 'QW3_METAL_KV_Q8_0': '0', 'QW3_METAL_NGL': '64',
            'QW3_METAL_DENSE_PREFILL_FAST': '1', 'QW3_METAL_DENSE_PREFILL_K64': '1',
            'QW3_METAL_DENSE_GDN_CACHE': '1', 'MTL_DEBUG_LAYER': '1'}
env.update(settings)
m = {'started_utc': now(), 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
     'model': str(model), 'model_size': stat_before[0], 'model_mtime_ns': stat_before[1],
     'scope': '36 guarded matrices including extra tail threadgroup; one reused ctx256 session with disk oracle; no eight-prefix/checkpoint suite',
     'source_sha256': sources, 'source_snapshot_paths': snapshots, 'binary_sha256': binaries,
     'production_binary_sha256': production, 'environment': settings, 'runs': [],
     'additional_weight_buffer_bytes': 0, 'additional_session_data_buffers': 0,
     'shader_changed': True, 'variants': {'1': 'rows_per_simd_2', '2': 'rows_per_simd_8'}, 'build_warning_free': True}


def save():
    (archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')


save()
try:
    tasks = [(f'matrix_v{v}', ['./qw3-iq3-rows-test', str(model), str(v)]) for v in [1, 2]]
    tasks.append(('model', ['./qw3-iq3-rows-model-test', str(model)]))
    for name, args in tasks:
        command = ['caffeinate', '-disu', *args]
        run = {'name': name, 'command': command, 'swap_before': swap()}
        start = time.monotonic()
        with (archive / (name + '.stdout')).open('w') as out, (archive / (name + '.stderr')).open('w') as err:
            result = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, timeout=300)
        run.update(exit_code=result.returncode, seconds=time.monotonic() - start, swap_after=swap())
        m['runs'].append(run)
        save()
        assert result.returncode == 0, name
        output = (archive / (name + '.stdout')).read_text()
        if name.startswith('matrix'):
            assert len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$', output, re.M)) == 18
            assert f'PASS iq3_rows={args[-1]} matrix_cases=18 finite_bit_exact=all' in output
        else:
            assert len(re.findall(r'^PASS iq3_rows_model .*sessions=1 additional_weight_buffer_bytes=0$', output, re.M)) == 3
        print('finished', name, 'exit', result.returncode, flush=True)
    m['all_passed'] = True
finally:
    m.update(source_unchanged=all(sha(root / p) == h for p, h in sources.items()),
             binary_unchanged=all(sha(root / p) == h for p, h in {**binaries, **production}.items()),
             model_unchanged=stat_before == (model.stat().st_size, model.stat().st_mtime_ns), completed_utc=now())
    save()
assert m['source_unchanged'] and m['binary_unchanged'] and m['model_unchanged']
print('numerical screening passed; native timing not yet run', flush=True)
