"""Full-vocabulary differential on shorter eligible tails using the actual partial kernels."""
from pathlib import Path
import json, os, re, subprocess, sys
out=Path(__file__).resolve().parent
root=out.parents[2]
work=Path(sys.argv[1]).resolve()/'partial'
old=out.parent/'staged128_partial_2026-11-17'
if sys.argv[2]=='build':
    subprocess.run(['python3',str(old/'prepare_frontier.py'),str(work)],cwd=root,check=True)
    sys.exit()
prefix=re.search(r'tokens=([0-9,]+)',(old/'frontier-v0.log').read_text()).group(1).split(',')
model=(root/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','MTL_','GGML_'))}
env.update(QW3_DENSE_METAL='1',QW3_METAL_NGL='64',QW3_METAL_KV_F16='1',QW3_DENSE_LLAMA_BATCH='1')
results=[]
for n in (64,65,87,95):
    for mode in (0,3):
        p=subprocess.run([str(work/'frontier'),str(model),*prefix[:n]],cwd=work,env={**env,'QW3_PARTIAL_STAGED':str(mode)},capture_output=True,text=True)
        output=p.stderr+p.stdout
        (out/f'frontier-{n}-v{mode}.log').write_text(output)
        m=re.search(r'top_match=(\d)/8 greedy_match=(\d)/8 rmse=([\d.e+-]+) max_abs=([\d.e+-]+)',output)
        if not m: raise RuntimeError(output[-1500:])
        results.append(dict(n=n,mode=mode,exit=p.returncode,top8=int(m[1]),greedy8=int(m[2]),rmse=float(m[3]),max_abs=float(m[4])))
        (out/'tail-frontier.json').write_text(json.dumps(results,indent=2)+'\n')
        print(results[-1],flush=True)

subprocess.run(['python3',str(out/'replica.py'),sys.argv[1]],check=True)

subprocess.run(['python3',str(out/'paired.py'),sys.argv[1],'run'],check=True)
