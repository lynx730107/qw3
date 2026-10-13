"""Sequential IQ3_S fused down/residual gate; no timing after a numerical failure."""
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
assert not (archive / 'manifest.json').exists(), 'Keep previous evidence; use a new archive for another run'
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
            ('gqa_barrier_2026-10-12', 'iq3_words_2026-10-11')]
manifests = [json.loads((p / 'manifest.json').read_text()) for p in previous]
paths = ['Makefile', '.gitignore', 'qw3.c', 'qw3.h', 'qw3_metal.m', 'qw3_metal.h',
    'qw3_bench.c', 'qw3_runtime_fingerprint.h', 'tools/runtime_fingerprint.py',
    *('metal/'+p.name for p in sorted((root/'metal').glob('*.metal'))),
    'tests/prepare_iq3_add.py', 'tests/iq3_add_hooks.inc', 'tests/test_iq3_add.m',
    'tests/test_iq3_add_model.m', 'tests/bench_iq3_add_native.m',
    'docs/measurements/iq3_input_2026-10-12/manifest.json',
    *('tests/.iq3_add/'+n for n in ('core.inc','core.diff','kernels.metal','model.inc','native.inc')),
    'docs/measurements/iq3_add_2026-10-13/run_screen.py',
    'docs/measurements/iq3_add_2026-10-13/run_native.py',
    'docs/measurements/iq3_add_2026-10-13/analyze_native.py']
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
    'qw3-iq3-add-test', 'qw3-iq3-add-model-test', 'qw3-iq3-add-native-bench')}
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
     'environment': settings, 'variants': {'1': 'fused residual expression', '2': 'fused residual explicit fma'},
     'scope': '288 synthetic guarded IQ3_S matrix fixtures comparing original matvec plus residual to fusion; one reused ctx256 session, prefix128 + greedy32; complete finite logits and canonical state; not eight-prefix/checkpoint integration gate',
     'additional_weight_buffer_bytes': 0, 'additional_session_data_buffers': 0,
     'threads_fixture': 64, 'threads_model': 64,
     'matrix_dynamic_threadgroup_bytes': 128, 'native_dynamic_threadgroup_bytes': 128,
     'build_warning_free': True, 'store_before': store_inventory(), 'runs': []}


def save():
    (archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')


save()
try:
    for v in (1, 2):
        name = f'matrix_v{v}'
        command = ['caffeinate', '-disu', './qw3-iq3-add-test', str(v)]
        run = {'name': name, 'command': command, 'swap_before': swap()}
        start = time.monotonic()
        with (archive / (name + '.stdout')).open('w') as out, (archive / (name + '.stderr')).open('w') as err:
            r = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, timeout=300)
        run.update(exit_code=r.returncode, seconds=time.monotonic() - start, swap_after=swap())
        run['matrix_cases_passed'] = len(re.findall(r'^matrix_exact .*guards=exact input=unchanged weights=unchanged$',
                                                   (archive / (name + '.stdout')).read_text(), re.M))
        m['runs'].append(run)
        save()
        print(name, 'exit', r.returncode, 'cases', run['matrix_cases_passed'], flush=True)
        if r.returncode == 0:
            assert run['matrix_cases_passed'] == 144
            assert f'PASS iq3_add={v} matrix_cases=144 finite_bit_exact=all' in (archive / (name + '.stdout')).read_text()
    matrices_passed = all(r['exit_code'] == 0 for r in m['runs'])
    if matrices_passed:
        name = 'model'
        command = ['caffeinate', '-disu', './qw3-iq3-add-model-test', str(model)]
        run = {'name': name, 'command': command, 'swap_before': swap()}
        start = time.monotonic()
        with (archive / 'model.stdout').open('w') as out, (archive / 'model.stderr').open('w') as err:
            r = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, timeout=300)
        run.update(exit_code=r.returncode, seconds=time.monotonic() - start, swap_after=swap())
        m['runs'].append(run)
        save()
        if r.returncode == 0:
            assert len(re.findall(r'^PASS iq3_add_model .*sessions=1 additional_weight_buffer_bytes=0$',
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
