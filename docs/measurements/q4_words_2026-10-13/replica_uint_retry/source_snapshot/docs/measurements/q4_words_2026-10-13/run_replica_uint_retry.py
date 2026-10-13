"""Independent six-pair AB/BA replica; original buffers and every sample retained."""
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
archive = screen / 'replica_uint_retry'
archive.mkdir(exist_ok=False)
old = json.loads((screen / 'manifest.json').read_text())
assert old['all_passed'] and old['source_unchanged'] and old['binary_unchanged']
model = Path(old['model'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def swap():
    return subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()


assert all(sha(root / p) == h for p, h in old['source_sha256'].items())
assert all(sha(root / p) == h for p, h in old['binary_sha256'].items())
assert (model.stat().st_size, model.stat().st_mtime_ns) == (old['model_size'], old['model_mtime_ns'])
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
settings = {k: v for k, v in old['environment'].items() if k != 'MTL_DEBUG_LAYER'}
settings['QW3_BENCH_PRINT_SAMPLES'] = '1'
env.update(settings)
m = {'started_utc': now(), 'revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(), 'initial_revision': old['revision'],
     'source_sha256': old['source_sha256'],
     'source_snapshot_paths': {p: '../' + s for p, s in old['source_snapshot_paths'].items()},
     'binary_sha256': {'qw3-q4-words-native-bench': old['binary_sha256']['qw3-q4-words-native-bench']},
     'production_binary_sha256': old['production_binary_sha256'],
     'model': str(model), 'model_size': old['model_size'], 'model_mtime_ns': old['model_mtime_ns'],
     'scope': 'ordinary single-session tg32 depth128 ctx256; original weight/session data buffers',
     'controller_power_command': ['caffeinate','-disu','python3','docs/measurements/q4_words_2026-10-13/run_replica_uint_retry.py'], 'context': 256, 'gpu_layers': 64, 'kv': 'f16', 'repetitions': 3, 'warmup': True,
     'cooldown_seconds_by_process': [180]+[60]*11, 'order': [0,2,2,0,0,2,2,0,0,2,2,0], 'runs': []}


# Freeze this controller separately; leave the numerical manifest untouched.
import shutil
script_name='docs/measurements/q4_words_2026-10-13/run_replica_uint_retry.py'
frozen=archive/'source_snapshot'/script_name
frozen.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(root/script_name,frozen)
m['source_sha256']={**m['source_sha256'],script_name:sha(root/script_name)}
m['source_snapshot_paths']={**m['source_snapshot_paths'],script_name:str(frozen.relative_to(archive))}
m['selection_reason']='Initial uint screen +1.337% vs adjacent controls, +1.915% / +0.760% against each; ushort has first-control-biased +2.516% with minimum +0.731%. Select uint for less distant adjacent controls and require independent AB/BA replication. No comparison of absolute rates across variants.'
m['initial_native_manifest_sha256']=sha(screen/'native/manifest.json')
m['initial_native_analysis_sha256']=sha(screen/'native/analysis.json')
m['previous_replica_manifest_sha256']=sha(screen/'replica_uint/manifest.json')
m['previous_replica_analysis_sha256']=sha(screen/'replica_uint/analysis.json')
m['retry_reason']='First uint replica +3.492% with all six pairs positive but OFF spread 6.572% exceeds 5%. Preserve all 36 samples; fresh complete balanced campaign, no selective reruns.'

def save():
    (archive / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')


save()
try:
    for index, variant in enumerate(m['order']):
        name = f'block_{index}_v{variant}'
        print('cooldown before', name, flush=True)
        time.sleep(m['cooldown_seconds_by_process'][index])
        local_env = {**env, 'QW3_PRIVATE_Q4_WORDS': str(variant)}
        command = ['caffeinate', '-disu', './qw3-q4-words-native-bench', '-m', str(model),
                   '--metal', '--ngl', '64', '--llama-style', '--ctx-alloc', '256',
                   '-p', '0', '-n', '32', '-d', '128', '-r', '3', '--seed', '1', '-t', '4']
        run = {'name': name, 'variant': variant, 'command': command,
               'environment': {**settings, 'QW3_PRIVATE_Q4_WORDS': str(variant)}, 'swap_before': swap()}
        start = time.monotonic()
        with (archive / (name + '.stdout')).open('w') as out, (archive / (name + '.stderr')).open('w') as err:
            result = subprocess.run(command, cwd=root, env=local_env, stdout=out, stderr=err, timeout=180)
        run.update(exit_code=result.returncode, seconds=time.monotonic() - start, swap_after=swap())
        log = (archive / (name + '.stderr')).read_text()
        run['samples'] = [{'repetition': int(n), 'tps': float(tps), 'ms': float(ms)}
                          for n, tps, ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms', log)]
        count = re.search(r'PASS q4_words_native variant=(\d+) calls=(\d+) additional_weight_buffer_bytes=0', log)
        run['candidate_calls'] = int(count.group(2)) if count else None
        m['runs'].append(run)
        save()
        assert result.returncode == 0 and count and int(count.group(1)) == variant
        assert len(run['samples']) == 3 and run['candidate_calls'] == (3492 if variant else 0)
        print('finished', name, 'mean_ms', statistics.mean(s['ms'] for s in run['samples']), flush=True)
finally:
    m.update(source_unchanged=all(sha(root / p) == h for p, h in m['source_sha256'].items()),
             binary_unchanged=all(sha(root / p) == h for p, h in {**m['binary_sha256'], **m['production_binary_sha256']}.items()),
             model_unchanged=(model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns']),
             completed_utc=now())
    save()
assert m['source_unchanged'] and m['binary_unchanged'] and m['model_unchanged']
print('native screening finished', flush=True)
