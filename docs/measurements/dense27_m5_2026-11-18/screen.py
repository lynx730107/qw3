"""Run all GPU diagnostics sequentially. Use fresh paths and retain each result."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

out=Path(__file__).resolve().parent
root=out.parents[2]
work=Path(sys.argv[1]).resolve()
model=str((root/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve())
base={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
base.update(QW3_METAL_KV_F16='1',QW3_METAL_KV_Q8_0='0',QW3_METAL_NGL='64',QW3_BENCH_PRINT_SAMPLES='1')
manifest=out/'screen.json'
rows=json.loads(manifest.read_text()) if manifest.exists() else []
def run(name,cmd,cwd=work,env=None,rest=0):
    existing=[r for r in rows if r['name']==name]
    if existing: return existing[-1]['exit']==0
    time.sleep(rest)
    print('starting',name,flush=True)
    start=time.monotonic()
    with (out/(name+'.log')).open('x') as log:
        p=subprocess.run(['caffeinate','-disu',*cmd],cwd=cwd,env={**base,**(env or {})},stdout=log,stderr=subprocess.STDOUT)
    rows.append(dict(name=name,cwd=str(cwd),command=cmd,env=env or {},exit=p.returncode,seconds=time.monotonic()-start,rest_before=rest))
    manifest.write_text(json.dumps(rows,indent=2)+'\n')
    print('finished',name,p.returncode,flush=True)
    return p.returncode==0

run('layer-screen',['python3',str(out/'layers.py'),str(work),'run'],work)
partial=work/'partial'
if run('tail-quality',['./quality',model,'3','8'],partial,{'MTL_DEBUG_LAYER':'1'}):
    for size in (63,64,95):
        for mode in (0,3):
            run(f'tail-route-{size}-v{mode}',['./trace','-m',model,'--metal','--ngl','64','--llama-style','--ctx-alloc','256','-p',str(size),'-n','0','-r','1','--seed','1','-t','4'],partial,{'QW3_PARTIAL_STAGED':str(mode)})
    for size in (65,95):
        run(f'tail-checkpoint-{size}',['./quality',model,'3','32',str(size)],partial)
    for cached,added in [(0,63),(0,64),(0,65),(0,69),(0,71),(0,87),(0,95),(0,96),(0,128),
                         (2404,727),(3250,709),(4101,705),(3976,991),(5230,967)]:
        run(f'turn-{cached}-{added}',['./turn',model,str(cached),str(added)],partial,rest=3)

for depth in (1024,4096,8192):
    run(f'profile-{depth}',['./profile_depth',model,str(depth)],rest=3)

def decode_screen(kind, env_key):
    for j,mode in enumerate((0,1,1,0,1,0,0,1)):
        ok=run(f'{kind}-tg-{j}-v{mode}',[f'./{kind}_native','-m',model,'--metal','--ngl','64','--llama-style',
               '--ctx-alloc','256','-p','0','-n','32','-d','128','-r','2','--seed','1','-t','4'],env={env_key:str(mode)},rest=8)
        if not ok: break
if run('gqa-model',['./gqa_model',model,'1'],env={'MTL_DEBUG_LAYER':'1'}):
    decode_screen('gqa','QW3_PRIVATE_GQA_HEADS_PROBE')
if run('nibble-matrix',['./nibble_matrix','1'],env={'MTL_DEBUG_LAYER':'1'}) and run('nibble-model',['./nibble_model',model],env={'MTL_DEBUG_LAYER':'1'}):
    decode_screen('nibble','QW3_PRIVATE_IQ3_NIBBLE')

# Confirm long-depth comparisons with comparable work immediately before tg32.
run('matched-controller',['python3',str(out/'depth.py'),str(work/'matched_bench'),'matched'])
