"""One reused ctx256 model gate for original shaders and an FFN scheduling window."""
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
binary = Path('/tmp/qw3-ffn-concurrent-model-test')
assert not (archive / 'manifest.json').exists()
assert not (archive / 'build.stderr').read_text()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def inventory():
    store = Path('/Users/pasqualecapasso/.qw3/agent_store')
    return {str(p.relative_to(store)): [p.stat().st_size, p.stat().st_mtime_ns, sha(p)]
            for p in store.rglob('*') if p.is_file()}


previous = root / 'docs/measurements/iq3_nibble_2026-10-14'
old = json.loads((previous / 'manifest.json').read_text())
paths = ['.gitignore', 'qw3.c', 'qw3.h', 'qw3_metal.m', 'qw3_metal.h', 'qw3_bench.c',
         'qw3_runtime_fingerprint.h', *('metal/' + p.name for p in sorted((root / 'metal').glob('*.metal'))),
         'tests/prepare_decode_ffn_concurrent.py', 'tests/test_decode_ffn_concurrent_model.m',
         'tests/bench_decode_ffn_concurrent_native.m',
         *('tests/.decode_ffn_concurrent/' + n for n in ('qw3_private.c', 'hooks.inc', 'model.inc', 'native.inc')),
         'docs/measurements/decode_ffn_concurrent_2026-10-14/run_model.py']
sources = {name: sha(root / name) for name in paths}
snapshots = {}
for name, digest in sources.items():
    if old['source_sha256'].get(name) == digest:
        frozen = (previous / old['source_snapshot_paths'][name]).resolve()
        assert sha(frozen) == digest
    else:
        frozen = archive / 'source_snapshot' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, frozen)
    snapshots[name] = os.path.relpath(frozen, archive)
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
settings = {'QW3_METAL_KV_F16': '1', 'QW3_METAL_KV_Q8_0': '0', 'QW3_METAL_NGL': '64',
            'QW3_METAL_DENSE_PREFILL_FAST': '1', 'QW3_METAL_DENSE_PREFILL_K64': '1',
            'QW3_METAL_DENSE_GDN_CACHE': '1', 'MTL_DEBUG_LAYER': '1'}
env.update(settings)
production = {n: sha(root / n) for n in ('qw3-cli', 'qw3-agent', 'qw3-bench')}
m = {'started_utc': now(), 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
     'source_sha256': sources, 'source_snapshot_paths': snapshots, 'binary': str(binary),
     'binary_sha256': sha(binary), 'production_binary_sha256': production,
     'model': str(model), 'model_size': model.stat().st_size, 'model_mtime_ns': model.stat().st_mtime_ns,
     'environment': settings, 'store_before': inventory(), 'command': ['caffeinate', '-disu', str(binary), str(model)],
     'scope': 'baseline/concurrent FFN fallback; one reused ctx256 session, prefix128 and greedy32, full finite vocabulary and canonical state; not checkpoint gate',
     'additional_weight_or_session_data_buffers': 0, 'metal_shaders_changed': False}
(archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
start = time.monotonic()
try:
    with (archive / 'model.stdout').open('w') as out, (archive / 'model.stderr').open('w') as err:
        r = subprocess.run(m['command'], cwd=root, env=env, stdout=out, stderr=err, timeout=300)
    m['exit_code'] = r.returncode
    phases = re.findall(r'^PASS ffn_concurrent_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$', (archive / 'model.stdout').read_text(), re.M)
    m['all_passed'] = r.returncode == 0 and phases == [('0', '0'), ('1', '992')]
finally:
    m.update(completed_utc=now(), seconds=time.monotonic() - start,
             source_unchanged=all(sha(root / n) == h for n, h in sources.items()),
             binary_unchanged=sha(binary) == m['binary_sha256'],
             production_unchanged=all(sha(root / n) == h for n, h in production.items()),
             model_unchanged=(model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns']),
             store_after=inventory())
    m['store_unchanged'] = m['store_before'] == m['store_after']
    (archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'production_unchanged', 'model_unchanged', 'store_unchanged'))
print('FFN concurrent numerical gate', 'PASS' if m['all_passed'] else 'FAIL', flush=True)
raise SystemExit(0 if m['all_passed'] else 1)
