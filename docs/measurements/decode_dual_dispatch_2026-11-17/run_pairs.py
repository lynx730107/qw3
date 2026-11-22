"""Check close-range OFF/ON comparisons in one reused model session."""
from pathlib import Path
from statistics import mean
import hashlib
import json
import os
import re
import subprocess
import sys

archive = Path(__file__).resolve().parent
work = Path(sys.argv[1]).resolve()
model = (Path.cwd()/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
mode_on = int(sys.argv[2]) if len(sys.argv)>2 else 1
assert mode_on in (1, 2)
stem = 'paired' if mode_on == 1 else 'paired-mixed'
env = {k: v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
env.update(QW3_METAL_KV_F16='1', QW3_BENCH_PRINT_SAMPLES='1', QW3_DUAL_DISPATCH=str(mode_on))
cmd = ['./paired_build/paired', '-m', str(model), '--metal', '--ngl', '64', '--llama-style',
       '--ctx-alloc', '256', '-p', '0', '-n', '32', '-d', '128', '-r', '8', '--seed', '1', '-t', '4']
files = ['paired_build/paired', 'paired_build/driver.m', 'paired_build/bench_private.c', 'adapter.inc',
         'qw3.c', 'qw3_metal.m', *[str(p.relative_to(work)) for p in sorted(work.glob('metal/*.metal'))]]
def hashes():
    return {p:hashlib.sha256((work/p).read_bytes()).hexdigest() for p in files}
before = hashes()
with (archive/f'{stem}.log').open('w') as f:
    p = subprocess.run(cmd, cwd=work, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
raw = (archive/f'{stem}.log').read_text()
check = re.findall(r'dual_pair rep=(\d+) mode=(\d+) same=(\d+) mixed=(\d+)', raw)
data = re.findall(r'sample (\d+)/8 [\d.]+ tok/s ([\d.]+) ms', raw)
order = [0,mode_on,mode_on,0,0,mode_on,mode_on,0]
assert p.returncode == 0 and before == hashes()
assert [tuple(map(int,x)) for x in check]==[
    (i+1,mode,384 if mode&1 else 0,800 if mode&2 else 0) for i,mode in enumerate(order)]
assert [int(i) for i,_ in data]==list(range(1,9))
samples = [float(ms) for _,ms in data]
gains=[]
for i in range(0,8,2):
    a,b = sorted(zip(order[i:i+2], samples[i:i+2]))
    gains.append(100*(a[1]/b[1]-1))
off,on = [mean(ms for mode,ms in zip(order,samples) if mode==v) for v in [0,mode_on]]
result = {'command':cmd, 'sha256':before, 'order':order, 'ms':samples,
          'off_tok_s':32000/off, 'on_tok_s':32000/on,
          'gain_pct':100*(off/on-1), 'paired_pct':gains}
(archive/f'{stem}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k.endswith(('pct','tok_s'))},indent=2))
