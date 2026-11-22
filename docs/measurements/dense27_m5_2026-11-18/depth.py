"""Ordinary tg32 by prefix depth, two independent pairs in reverse order."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
repo = out.parents[2]
bench = Path(sys.argv[1]).resolve()
llama = (repo / '../../llama.cpp/build/bin/llama-bench').resolve()
model = (repo / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
env = {k:v for k,v in os.environ.items() if not k.startswith(('QW3_', 'MTL_', 'GGML_'))}
env.update(QW3_METAL_KV_F16='1', QW3_METAL_KV_Q8_0='0', QW3_BENCH_PRINT_SAMPLES='1')
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

manifest = dict(binary_hashes={str(p):digest(p) for p in (bench,llama)},
                sources={str(p.relative_to(repo)):digest(p) for p in
                         [repo/'qw3.c', repo/'qw3_metal.m', repo/'qw3_bench.c', *sorted((repo/'metal').glob('*.metal'))]},
                model=str(model), model_size=model.stat().st_size, model_mtime_ns=model.stat().st_mtime_ns,
                llama_head=subprocess.check_output(['git','-C',str(llama.parents[2]),'rev-parse','HEAD'],text=True).strip(),
                synthetic_streams_differ=True, cooldown_seconds=8, runs=[])
resume8 = sys.argv[2:] == ['matched8']
matched = resume8 or sys.argv[2:] == ['matched']
manifest['warmup'] = 'both one token at empty context' if matched else 'ordinary engine-specific warmups'
if matched:
    out=out/('matched8' if resume8 else 'matched')
    out.mkdir()
results=out/'depth.json'
if results.exists():
    raise FileExistsError(results)
for depth in ((8192,) if resume8 else (4096,8192) if matched else (1024,4096,8192)):
    for order,engine in enumerate(('qw3','llama','llama','qw3')):
        name=f'd{depth}-{order}-{engine}'
        args = ([str(bench),'--metal','--ngl','64','--llama-style','--ctx-alloc',str(((depth+32+255)//256)*256),'--seed','1'] if engine=='qw3' else
                [str(llama),'-ngl','999','-ctk','f16','-ctv','f16','-fa','on','-b','128','-ub','128','-o','json','-v'])
        args += ['-m',str(model),'-p','0','-n','32','-d',str(depth),'-r','1','-t','4']
        time.sleep(8)
        start=time.monotonic()
        swap=lambda:subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip()
        before=swap()
        with (out/(name+'.stdout')).open('x') as so,(out/(name+'.stderr')).open('x') as se:
            p=subprocess.run(['caffeinate','-disu',*args],env=env,cwd=repo,stdout=so,stderr=se)
        row=dict(depth=depth, order=order, engine=engine, args=args, exit=p.returncode, seconds=time.monotonic()-start,swap_before=before,swap_after=swap())
        manifest['runs'].append(row)
        results.write_text(json.dumps(manifest,indent=2)+'\n')
        if p.returncode: raise RuntimeError(name)
        output=(out/(name+'.stdout')).read_text()
        if engine=='llama':
            data=json.loads(output)[0]
            assert (data['n_depth'],data['n_gen'])==(depth,32)
            row.update(tps=data['samples_ts'][0],ms=data['samples_ns'][0]/1e6)
        else:
            data=list(csv.DictReader(output.splitlines()))[0]
            assert int(data['n_depth'])==depth and int(data['n_gen'])==32
            row.update(tps=float(data['avg_tps']),ms=float(data['avg_ms']),session_bytes=int(data['session_bytes']))
        results.write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,row['tps'],flush=True)
