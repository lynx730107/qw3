"""Ordinary tg32/depth128 screening; cooldown before every process."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import statistics
import subprocess
import time

root = Path(__file__).resolve().parents[3]
screen = Path(__file__).resolve().parent
archive = screen / 'selected_native'
archive.mkdir(exist_ok=False)
old_dir=screen/'selected_v2_corrected'
old = json.loads((old_dir / 'manifest.json').read_text())
assert old['selected_variant']==2
assert old['all_passed'] and old['source_unchanged'] and old['binary_unchanged']
model = Path(old['model'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def swap():
    return subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()


assert all(sha(root / p) == h for p, h in old['source_sha256'].items())
assert sha(Path(old['binary'])) == old['binary_sha256']
assert (model.stat().st_size, model.stat().st_mtime_ns) == (old['model_size'], old['model_mtime_ns'])
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
settings = {k: v for k, v in old['environment'].items() if k != 'MTL_DEBUG_LAYER'}
settings['QW3_BENCH_PRINT_SAMPLES'] = '1'
env.update(settings)
m = {'started_utc': now(), 'revision': old['revision'],
     'source_sha256': old['source_sha256'],
     'source_snapshot_paths': {p: os.path.relpath((old_dir/s).resolve(),archive) for p,s in old['source_snapshot_paths'].items()},
     'binary_sha256': {'/tmp/qw3-gqa-groups-native-bench': sha(Path('/tmp/qw3-gqa-groups-native-bench'))},
     'production_binary_sha256': old['production_binary_sha256'],
     'model': str(model), 'model_size': old['model_size'], 'model_mtime_ns': old['model_mtime_ns'],
     'scope': 'ordinary single-session tg32 depth128 ctx256; original weight/session data buffers',
     'controller_power_command': ['caffeinate','-disu','python3','docs/measurements/gqa_groups_2026-10-14/run_selected_native.py'], 'context': 256, 'gpu_layers': 64, 'kv': 'f16', 'repetitions': 3, 'warmup': True,
     'cooldown_seconds_before_every_process': 60, 'order': [0, 2, 0, 2, 0], 'runs': []}


# Freeze the timing controller separately from the completed numerical gate.
import shutil
script_name='docs/measurements/gqa_groups_2026-10-14/run_selected_native.py'
frozen=archive/'source_snapshot'/script_name
frozen.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(root/script_name,frozen)
m['source_sha256']={**m['source_sha256'],script_name:sha(root/script_name)}
m['source_snapshot_paths']={**m['source_snapshot_paths'],script_name:str(frozen.relative_to(archive))}


def save():
    (archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')


save()
try:
    for index, variant in enumerate(m['order']):
        name = f'block_{index}_v{variant}'
        print('cooldown before', name, flush=True)
        time.sleep(60)
        local_env = {**env, 'QW3_PRIVATE_GQA_GROUPS': str(variant)}
        command = ['caffeinate', '-disu', '/tmp/qw3-gqa-groups-native-bench', '-m', str(model),
                   '--metal', '--ngl', '64', '--llama-style', '--ctx-alloc', '256',
                   '-p', '0', '-n', '32', '-d', '128', '-r', '3', '--seed', '1', '-t', '4']
        run = {'name': name, 'variant': variant, 'command': command,
               'environment': {**settings, 'QW3_PRIVATE_GQA_GROUPS': str(variant)}, 'swap_before': swap()}
        start = time.monotonic()
        with (archive / (name + '.stdout')).open('w') as out, (archive / (name + '.stderr')).open('w') as err:
            result = subprocess.run(command, cwd=root, env=local_env, stdout=out, stderr=err, timeout=180)
        run.update(exit_code=result.returncode, seconds=time.monotonic() - start, swap_after=swap())
        log = (archive / (name + '.stderr')).read_text()
        run['samples'] = [{'repetition': int(n), 'tps': float(tps), 'ms': float(ms)}
                          for n, tps, ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms', log)]
        count = re.search(r'PASS gqa_groups_native variant=(\d+) calls=(\d+) additional_weight_buffer_bytes=0', log)
        run['candidate_calls'] = int(count.group(2)) if count else None
        m['runs'].append(run)
        save()
        assert result.returncode == 0 and count and int(count.group(1)) == variant
        assert len(run['samples']) == 3 and run['candidate_calls'] == (1552 if variant else 0)
        print('finished', name, 'mean_ms', statistics.mean(s['ms'] for s in run['samples']), flush=True)
finally:
    m.update(source_unchanged=all(sha(root / p) == h for p, h in m['source_sha256'].items()),
             binary_unchanged=all(sha(root / p) == h for p, h in {**m['binary_sha256'], **m['production_binary_sha256']}.items()),
             model_unchanged=(model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns']),
             completed_utc=now())
    save()
assert m['source_unchanged'] and m['binary_unchanged'] and m['model_unchanged']
print('native screening finished', flush=True)
