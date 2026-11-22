"""Sequential Metal prefill measurements. Run from the repository root."""
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

work = Path(sys.argv[1]).resolve()
series = sys.argv[2]
archive = Path(__file__).resolve().parent
model = (Path.cwd() / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('QW3_', 'GGML_', 'MTL_'))}
env.update(QW3_METAL_KV_F16='1', QW3_BENCH_PRINT_SAMPLES='1')
if series == 'screen':
    blocks = {127: [0, 1, 2, 2, 1, 0], 96: [2, 1, 0, 0, 1, 2]}
    cooldown = 5
    repetitions = 3
elif series == 'replica':
    mode = int(sys.argv[3])
    if mode not in (1, 2, 3):
        raise ValueError('Expected replica mode 1, 2 or 3')
    blocks = {127: [0, mode, mode, 0, 0, mode, mode, 0],
              96: [mode, 0, 0, mode],
              128: [0, mode, mode, 0]}
    cooldown = 8
    repetitions = 3
elif series == 'formats':
    blocks = {127: [2, 3, 3, 2], 96: [3, 2, 2, 3]}
    cooldown = 5
    repetitions = 3
else:
    raise ValueError(series)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity():
    files = [work / 'bench', *work.glob('*.h'), *work.glob('*.m'),
             *work.glob('*.c'), *work.glob('metal/*.metal')]
    return {str(p.relative_to(work)): digest(p) for p in sorted(files)}

def model_stat():
    s = model.stat()
    return [s.st_size, s.st_mtime_ns]

manifest = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'model': str(model), 'model_stat': model_stat(), 'sha256': identity(),
            'cooldown_seconds': cooldown, 'order': blocks, 'runs': [],
            'started': datetime.now(timezone.utc).isoformat()}
result_file = archive / f'{series}.json'

def save():
    result_file.write_text(json.dumps(manifest, indent=2) + '\n')

save()
pattern = rf'sample (\d+)/{repetitions} [\d.]+ tok/s ([\d.]+) ms'
for n, order in blocks.items():
    for i, mode in enumerate(order):
        time.sleep(cooldown)
        filename = f'{series}-{n}-{i}-v{mode}.log'
        command = ['./bench', '-m', str(model), '--metal', '--ngl', '64',
                   '--llama-style', '--ctx-alloc', '256', '-p', str(n),
                   '-n', '0', '-r', str(repetitions), '--seed', '1', '-t', '4']
        with (archive / filename).open('w') as log:
            process = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(env, QW3_PARTIAL_STAGED=str(mode)), timeout=120)
        samples = re.findall(pattern, (archive / filename).read_text())
        run = {'n': n, 'mode': mode, 'command': command, 'log': filename,
               'exit': process.returncode, 'ms': [float(ms) for _, ms in samples]}
        manifest['runs'].append(run)
        save()
        if process.returncode or [int(rep) for rep, _ in samples] != list(range(1, repetitions + 1)):
            raise RuntimeError(f'Incomplete run {filename}')
        run.update(mean_ms=mean(run['ms']), sd_ms=stdev(run['ms']))
        save()
        print(n, mode, round(run['mean_ms'], 3), flush=True)
assert manifest['sha256'] == identity() and manifest['model_stat'] == model_stat()
manifest['complete'] = True
save()
