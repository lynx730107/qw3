"""Benchmark ordinary 32-token generation in balanced separate processes."""
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
import hashlib
import json
import os
import re
import subprocess
import sys
import time

archive = Path(__file__).resolve().parent
work = Path(sys.argv[1]).resolve()
model = (Path.cwd() / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
series = sys.argv[2]
if series == 'screen':
    order = [0, 1, 0, 2, 0, 3, 0, 3, 0, 2, 0, 1, 0]
    cooldown = 6
elif series == 'replica':
    mode = int(sys.argv[3])
    assert mode in (1, 2, 3)
    order = [0, mode, mode, 0, 0, mode, mode, 0]
    cooldown = 12
else:
    raise ValueError(series)
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
env.update(QW3_METAL_KV_F16='1', QW3_BENCH_PRINT_SAMPLES='1')
command = ['./bench', '-m', str(model), '--metal', '--ngl', '64', '--llama-style',
           '--ctx-alloc', '256', '-p', '0', '-n', '32', '-d', '128', '-r', '3',
           '--seed', '1', '-t', '4']
def identity():
    return {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [work/'bench', work/'quality', *sorted(work.glob('qw3*')),
                      *sorted(work.glob('metal/*.metal')), work/'bench.m', work/'adapter.inc', work/'quality.m']}

def model_stat():
    stat = model.stat()
    return [stat.st_size, stat.st_mtime_ns]

quality = (archive/'quality-api.log').read_text()
assert 'PASS length=128 mode=3 continuation=32 checkpoint=exact' in quality
revision = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
manifest = {'command': command, 'revision': revision, 'model_stat': model_stat(),
            'sha256': identity(), 'order': order, 'cooldown_seconds': cooldown,
            'started_utc': datetime.now(timezone.utc).isoformat(), 'runs': []}

def save():
    (archive/f'{series}.json').write_text(json.dumps(manifest, indent=2)+'\n')

save()
for i, mode in enumerate(order):
    time.sleep(cooldown)
    name = f'{series}-{i:02d}-v{mode}.log'
    with (archive/name).open('w') as f:
        result = subprocess.run(command, cwd=work, env=dict(env, QW3_DUAL_DISPATCH=str(mode)),
                                stdout=f, stderr=subprocess.STDOUT, timeout=150)
    output = (archive/name).read_text()
    samples = [float(ms) for ms in re.findall(r'sample \d+/3 [\d.]+ tok/s ([\d.]+) ms', output)]
    calls = re.search(r'dual_dispatch mode=(\d+) same=(\d+) mixed=(\d+)', output)
    record = {'mode': mode, 'log': name, 'exit': result.returncode, 'ms': samples,
              'counts': tuple(map(int,calls.groups())) if calls else None}
    manifest['runs'].append(record)
    save()
    assert result.returncode == 0 and len(samples) == 3 and calls
    assert record['counts'] == (mode, 97*12 if mode&1 else 0, 97*25 if mode&2 else 0)
    record.update(mean_ms=mean(samples), sd_ms=stdev(samples))
    print(i, mode, round(record['mean_ms'],3), record['counts'], flush=True)
    save()

assert manifest['sha256']==identity() and manifest['model_stat']==model_stat()
runs = manifest['runs']
comparisons = []
for i, run in enumerate(runs):
    if not run['mode']:
        continue
    if series == 'screen':
        neighbors = [i-1, i+1]
    else:
        neighbors = [i+1 if i%2==0 else i-1]
    assert all(runs[j]['mode']==0 for j in neighbors)
    base = mean(runs[j]['mean_ms'] for j in neighbors)
    comparisons.append({'index': i, 'mode': run['mode'], 'off_indices': neighbors,
                        'off_ms': base, 'saved_ms': base-run['mean_ms'],
                        'throughput_gain_pct': 100*(base/run['mean_ms']-1)})
manifest.update(completed=True, comparisons=comparisons)
save()
