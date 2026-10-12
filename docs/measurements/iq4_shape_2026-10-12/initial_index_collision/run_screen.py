"""Sequential bit-exact gates before any private unroll timing."""
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
model = Path('/Users/pasqualecapasso/Downloads/models/Qwen3.8-27B-UD-Q3_K_XL.gguf')
assert not (archive / 'build.stderr').read_text()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def swap():
    return subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()


def store_inventory():
    store = Path('/Users/pasqualecapasso/.qw3/agent_store')
    return {str(p.relative_to(store)): {'bytes': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha(p)}
            for p in sorted(store.rglob('*')) if p.is_file()}


previous = [root / 'docs/measurements' / d for d in
            ('agent_context_pressure_2026-10-11', 'iq4_unroll_2026-10-12')]
manifests = [json.loads((p / 'manifest.json').read_text()) for p in previous]
paths = list(manifests[0]['source_sha256']) + [
    'tests/prepare_iq4_shape.py', 'tests/test_iq4_shape.m',
    'tests/test_iq4_shape_model.m', 'tests/bench_iq4_shape_native.m',
    *('tests/.iq4_unroll/' + n for n in ('hooks.inc', 'matrix.inc', 'model.inc', 'native.inc')),
    *('tests/.iq4_shape/' + n for n in ('hooks.inc', 'kernels.metal', 'matrix.inc', 'model.inc', 'native.inc')),
    'docs/measurements/iq4_shape_2026-10-12/run_screen.py',
    'docs/measurements/iq4_shape_2026-10-12/run_native.py',
]
sources = {p: sha(root / p) for p in paths}
snapshots = {}
for name, digest in sources.items():
    frozen = None
    for directory, old in zip(previous, manifests):
        if old['source_sha256'].get(name) == digest:
            frozen = (directory / old['source_snapshot_paths'][name]).resolve()
            assert sha(frozen) == digest
            break
    if frozen is None:
        frozen = archive / 'source_snapshot' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, frozen)
    snapshots[name] = os.path.relpath(frozen, archive)
binaries = {n: sha(root / n) for n in (
    'qw3-iq4-shape-test', 'qw3-iq4-shape-model-test', 'qw3-iq4-shape-native-bench')}
production = {n: sha(root / n) for n in ('qw3-cli', 'qw3-agent', 'qw3-bench')}
model_stat = (model.stat().st_size, model.stat().st_mtime_ns)
settings = {'QW3_METAL_KV_F16': '1', 'QW3_METAL_KV_Q8_0': '0',
            'QW3_METAL_NGL': '64', 'QW3_METAL_DENSE_PREFILL_FAST': '1',
            'QW3_METAL_DENSE_PREFILL_K64': '1', 'QW3_METAL_DENSE_GDN_CACHE': '1',
            'MTL_DEBUG_LAYER': '1'}
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
env.update(settings)
m = {'started_utc': now(), 'revision': subprocess.check_output(
     ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
     'source_sha256': sources, 'source_snapshot_paths': snapshots,
     'binary_sha256': binaries, 'production_binary_sha256': production,
     'model': str(model), 'model_size': model_stat[0], 'model_mtime_ns': model_stat[1],
     'environment': settings, 'variants': {'1': 'constant K count', '2': 'constant K count and row stride'},
     'scope': '108 guarded matrices; one reused ctx256 session, prefix128 + greedy32; complete finite logits and canonical state; not eight-prefix/checkpoint integration gate',
     'additional_weight_buffer_bytes': 0, 'additional_session_data_buffers': 0,
     'threads': 64, 'rows_per_simd': 2, 'simdgroups': 2,
     'threadgroup_lut_bytes': 64, 'dynamic_threadgroup_bytes': 128,
     'build_warning_free': True, 'store_before': store_inventory(), 'runs': []}


def save():
    (archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')


save()
try:
    for v in (1, 2):
        name = f'matrix_v{v}'
        command = ['caffeinate', '-disu', './qw3-iq4-shape-test', str(model), str(v)]
        run = {'name': name, 'command': command, 'swap_before': swap()}
        start = time.monotonic()
        with (archive / (name + '.stdout')).open('w') as out, (archive / (name + '.stderr')).open('w') as err:
            r = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, timeout=300)
        run.update(exit_code=r.returncode, seconds=time.monotonic() - start, swap_after=swap())
        run['matrix_cases_passed'] = len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$',
                                                   (archive / (name + '.stdout')).read_text(), re.M))
        m['runs'].append(run)
        save()
        print(name, 'exit', r.returncode, 'cases', run['matrix_cases_passed'], flush=True)
        if r.returncode == 0:
            assert run['matrix_cases_passed'] == 54
    matrices_passed = all(r['exit_code'] == 0 for r in m['runs'])
    if matrices_passed:
        name = 'model'
        command = ['caffeinate', '-disu', './qw3-iq4-shape-model-test', str(model)]
        run = {'name': name, 'command': command, 'swap_before': swap()}
        start = time.monotonic()
        with (archive / 'model.stdout').open('w') as out, (archive / 'model.stderr').open('w') as err:
            r = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, timeout=300)
        run.update(exit_code=r.returncode, seconds=time.monotonic() - start, swap_after=swap())
        m['runs'].append(run)
        save()
        if r.returncode == 0:
            assert len(re.findall(r'^PASS iq4_shape_model .*sessions=1 additional_weight_buffer_bytes=0$',
                                  (archive / 'model.stdout').read_text(), re.M)) == 3
        print(name, 'exit', r.returncode, flush=True)
    m['all_passed'] = matrices_passed and m['runs'][-1]['name'] == 'model' and m['runs'][-1]['exit_code'] == 0
finally:
    m.update(source_unchanged=all(sha(root / p) == h for p, h in sources.items()),
             binary_unchanged=all(sha(root / p) == h for p, h in {**binaries, **production}.items()),
             model_unchanged=model_stat == (model.stat().st_size, model.stat().st_mtime_ns),
             store_after=store_inventory(), completed_utc=now())
    m['store_unchanged'] = m['store_before'] == m['store_after']
    save()
assert m['source_unchanged'] and m['binary_unchanged'] and m['model_unchanged'] and m['store_unchanged']
print('numerical gate', 'PASS' if m['all_passed'] else 'FAIL; no timing authorized', flush=True)
raise SystemExit(0 if m['all_passed'] else 1)
