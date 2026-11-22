"""Run from the checkout root, passing the build directory printed by prepare.py."""
from pathlib import Path
import hashlib, json, os, re, statistics, subprocess, sys, time
out=Path(__file__).resolve().parent
bench=Path(sys.argv[1])/'bench'
model=Path('../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
env.update(QW3_METAL_KV_F16='1',QW3_METAL_KV_Q8_0='0',QW3_METAL_NGL='64',QW3_BENCH_PRINT_SAMPLES='1')
args=[str(bench),'-m',str(model),'--metal','--ngl','64','--llama-style','--ctx-alloc','256','-p','0','-n','32','-d','128','-r','3','--seed','1','-t','4']
files=[Path(n) for n in ('qw3.c','qw3.h','qw3_metal.m','qw3_metal.h','qw3_bench.c')]+list(Path('metal').glob('*.metal'))
def hashes():
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes()
assert 'PASS projection_barriers_model phase=1' in (out/'quality.stdout').read_text()
runs=[]
for i,mode in enumerate([0,1,1,0,0,1,1,0]):
 time.sleep(15)
 with (out/f'{i}-{mode}.log').open('w') as log:
  subprocess.run(args,env=dict(env,QW3_PRIVATE_PROJECTION_BARRIERS=str(mode)),stdout=log,stderr=subprocess.STDOUT,check=True)
 raw=log.name.read_text() if isinstance(log.name,Path) else Path(log.name).read_text()
 ms=[float(v) for v in re.findall(r'sample \d+/3 [\d.]+ tok/s ([\d.]+) ms',raw)]
 assert len(ms)==3 and f'PASS projection_barriers_native variant={mode} calls={7663 if mode else 0} ' in raw
 runs.append(dict(mode=mode,ms=ms,mean=statistics.mean(ms)))
 print(i, mode,runs[-1]['mean'],flush=True)
assert before==hashes()
pairs=[]
for i in range(0,len(runs),2):
 off,on=sorted(runs[i:i+2],key=lambda x:x['mode'])
 pairs.append((off['mean']/on['mean']-1)*100)
data=dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),hashes=before,benchmark_sha256=hashlib.sha256(bench.read_bytes()).hexdigest(),args=args,rest_seconds=15,runs=runs,paired_percent=pairs)
(out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
print(pairs)
