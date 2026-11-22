"""Independent process pairs on the lower and upper tail, preserving all repetitions."""
from pathlib import Path
from statistics import mean
import json
import os
import re
import subprocess
import sys
import time

out=Path(__file__).resolve().parent
root=out.parents[2]
work=Path(sys.argv[1]).resolve()/'partial'
env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
env.update(QW3_METAL_KV_F16='1',QW3_METAL_KV_Q8_0='0',QW3_BENCH_PRINT_SAMPLES='1')
model=str((root/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve())
rows=[]
for n,order in [(65,(0,3,3,0)),(95,(3,0,0,3))]:
    for i,mode in enumerate(order):
        time.sleep(8)
        name=f'replica-{n}-{i}-v{mode}'
        cmd=['./bench','-m',model,'--metal','--ngl','64','--llama-style','--ctx-alloc','256','-p',str(n),'-n','0','-r','3','--seed','1','-t','4']
        with (out/(name+'.log')).open('x') as log:
            p=subprocess.run(['caffeinate','-disu',*cmd],cwd=work,env={**env,'QW3_PARTIAL_STAGED':str(mode)},stdout=log,stderr=subprocess.STDOUT)
        samples=re.findall(r'qw3-bench: sample (\d+)/3 .*? ([\d.]+) ms',(out/(name+'.log')).read_text())
        assert p.returncode==0 and [i for i,_ in samples]==['1','2','3']
        rows.append(dict(n=n,mode=mode,order=i,ms=[float(v) for _,v in samples],mean_ms=mean(float(v) for _,v in samples),command=cmd))
        (out/'tail-replica.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(name,rows[-1]['mean_ms'],flush=True)
