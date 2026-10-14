"""Close-pair diagnostic using the original timer and one original session."""
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
base = root / 'docs/measurements/decode_projection_barriers_2026-10-14'
old = json.loads((base / 'manifest.json').read_text())
assert old['all_passed'] and not (archive / 'manifest.json').exists()
assert not (archive / 'build.stderr').read_text()
binary = Path('/tmp/qw3-projection-pairs-bench')
model = Path(old['model'])


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


paths = ['.gitignore', 'qw3.c', 'qw3.h', 'qw3_metal.m', 'qw3_metal.h', 'qw3_bench.c',
         'qw3_runtime_fingerprint.h', *('metal/' + p.name for p in sorted((root / 'metal').glob('*.metal'))),
         'tests/prepare_projection_pairs.py', 'tests/bench_projection_pairs.m',
         *('tests/.projection_pairs/' + n for n in ('hooks.inc', 'qw3_private.c', 'qw3_private_metal.m', 'bench_private.c', 'native.inc')),
         'docs/measurements/projection_pairs_2026-10-14/run_pairs.py']
sources = {n: sha(root / n) for n in paths}
snapshots = {}
for name, digest in sources.items():
    if old['source_sha256'].get(name) == digest:
        frozen = (base / old['source_snapshot_paths'][name]).resolve()
        assert sha(frozen) == digest
    else:
        frozen = archive / 'source_snapshot' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, frozen)
    snapshots[name] = os.path.relpath(frozen, archive)
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
settings = {k: v for k, v in old['environment'].items() if k != 'MTL_DEBUG_LAYER'}
settings['QW3_BENCH_PRINT_SAMPLES'] = '1'
env.update(settings)
command = ['caffeinate', '-disu', str(binary), '-m', str(model), '--metal', '--ngl', '64',
           '--llama-style', '--ctx-alloc', '256', '-p', '0', '-n', '32', '-d', '128',
           '-r', '12', '--seed', '1', '-t', '4']
m = {'started_utc': now(), 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
     'source_sha256': sources, 'source_snapshot_paths': snapshots,
     'binary': str(binary), 'binary_sha256': sha(binary),
     'production_binary_sha256': {n: sha(root / n) for n in ('qw3-cli', 'qw3-agent', 'qw3-bench')},
     'model': str(model), 'model_size': model.stat().st_size, 'model_mtime_ns': model.stat().st_mtime_ns,
     'order': [0,1,1,0,0,1,1,0,0,1,1,0], 'environment': settings, 'command': command,
     'cooldown_before_process_seconds': 60,
     'scope': 'diagnostic close pairs, one process/engine/session; original reset/prefill/timed loops, one baseline warmup; not independent process replication or a llama comparison',
     'swap_before': subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()}
(archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
print('cooldown before close-pair diagnostic', flush=True)
time.sleep(60)
start = time.monotonic()
try:
    with (archive / 'pairs.stdout').open('w') as out, (archive / 'pairs.stderr').open('w') as err:
        r = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, timeout=180)
    m['exit_code'] = r.returncode
    log = (archive / 'pairs.stderr').read_text()
    m['samples'] = [{'repetition': int(n), 'tps': float(t), 'ms': float(ms)}
                    for n, t, ms in re.findall(r'qw3-bench: sample (\d+)/12 ([\d.]+) tok/s ([\d.]+) ms', log)]
    assert r.returncode == 0 and len(m['samples']) == 12
    assert 'PASS projection_pairs runs=12 sessions=1 additional_weight_buffer_bytes=0' in log
finally:
    m.update(completed_utc=now(), seconds=time.monotonic() - start,
             source_unchanged=all(sha(root / n) == h for n, h in sources.items()),
             binary_unchanged=sha(binary) == m['binary_sha256'],
             production_unchanged=all(sha(root / n) == h for n, h in m['production_binary_sha256'].items()),
             model_unchanged=(model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns']),
             swap_after=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip())
    (archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'production_unchanged', 'model_unchanged'))
print('close-pair diagnostic completed', flush=True)
