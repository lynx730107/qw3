#!/usr/bin/env python3
"""Frozen, sequential eight-task comparison; no model processes overlap.

Outputs raw logs, versioned inputs, immutable checks, pp/tg timing, process
physical footprint (Darwin), RSS and machine-wide swap samples. The evaluator
runs after the agent exits, with no opportunity to edit the authoritative check.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / 'tests/fixtures/hybrid_pilot.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def swap_bytes():
    text = subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True)
    m = re.search(r'used = ([0-9.]+)([MG])', text)
    return int(float(m[1]) * (2**20 if m[2]=='M' else 2**30)) if m else None

class Meter:
    def __init__(self):
        self.lib = ctypes.CDLL('/usr/lib/libproc.dylib') if os.uname().sysname == 'Darwin' else None
    def sample(self, parent):
        rows = subprocess.check_output(['ps','-axo','pid=,ppid=,rss='], text=True)
        table = {int(p): (int(pp),int(rss)*1024) for p,pp,rss in (r.split() for r in rows.splitlines())}
        pids = {parent}
        while True:
            extra = {p for p,(pp,rss) in table.items() if pp in pids}
            if extra <= pids: break
            pids |= extra
        usage = []
        for pid in sorted(pids):
            if pid not in table: continue
            physical = None
            if self.lib:
                buf=ctypes.create_string_buffer(256)
                if self.lib.proc_pid_rusage(pid,0,ctypes.byref(buf))==0:
                    physical=struct.unpack_from('10Q',buf,16)[7]
            usage.append(dict(pid=pid,ppid=table[pid][0],rss=table[pid][1],physical=physical))
        return dict(processes=usage,rss=sum(p['rss'] for p in usage),
                    physical=sum(p['physical'] or 0 for p in usage),swap=swap_bytes())

def timing(path):
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try: event=json.loads(line)
            except json.JSONDecodeError: continue
            if event['event'] in ('inference','hybrid_inference'): rows.append(event)
    return dict(prefill_tokens=sum(r.get('prefill_tokens',0) for r in rows),
                prefill_ms=sum(r.get('prefill_ms',0) for r in rows),
                generation_tokens=sum(r.get('generation_tokens',0) for r in rows),
                generation_ms=sum(r.get('generation_ms',0) for r in rows))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',default='all')
    parser.add_argument('--modes',default='moe,dense,hybrid')
    parser.add_argument('--cooldown',type=float,default=30)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 0<=args.cooldown<=60: parser.error('cooldown must be 0..60')
    binary=ROOT/'qw3-agent'
    moe=(ROOT/'../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf').resolve(strict=True)
    dense=(ROOT/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
    fixtures=json.loads(FIXTURES.read_text())
    cases=fixtures['cases']
    if args.cases!='all': cases=[c for c in cases if c['id'] in args.cases.split(',')]
    modes=args.modes.split(',')
    if not cases or any(m not in ('moe','dense','hybrid') for m in modes): parser.error('invalid cases/modes')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    manifest=args.output/'results.json'
    if args.resume and manifest.exists():
        results=json.loads(manifest.read_text())
        if results['binary_sha256']!=digest(binary) or results['fixture_sha256']!=digest(FIXTURES):
            parser.error('cannot mix different binaries or frozen fixtures in one evaluation')
    else:
        shutil.copyfile(FIXTURES,args.output/'fixtures.json')
        results=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            fixture_sha256=digest(FIXTURES),binary_sha256=digest(binary),
            models=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (moe,dense)],
            device=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            ram_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
            ctx=8192,kv='f16',offload='full',temperature=0,seed=1,
            moe_thinking='none',dense_thinking='on',dense_total_tokens=8192,
            moe_assistant_tokens=2048,dense_assistant_tokens=8192,tool_rounds=12,expert_seconds=1200,
            cooldown_seconds=args.cooldown,runs=[])
    def save(): manifest.write_text(json.dumps(results,indent=2)+'\n')
    save(); meter=Meter()
    for case in cases:
        for mode in modes:
            if any(r['case']==case['id'] and r['mode']==mode for r in results['runs']): continue
            run_dir=args.output/(case['id']+'-'+mode); run_dir.mkdir()
            work=run_dir/'work';work.mkdir()
            (work/'README.md').write_text(case['spec']+'\n')
            (work/'solution.py').write_text(case['source'])
            (work/'check.py').write_text(case['check']+"\nprint('CHECKS_OK')\n")
            before=digest(work/'check.py')
            prompt=('Fix solution.py to satisfy README.md completely. Preserve the API. Do not change README.md or check.py. '
                    'Read the specification and inspect solution.py using a precise read range or get_function. '
                    'Make the changes, run python3 check.py using bash, and report verified results and any unresolved issues. '
                    'The listed tests are examples: handle all edge cases specified. Be concise.')
            (run_dir/'prompt.txt').write_text(prompt)
            command=[str(binary),'--metal','-m',str(dense if mode=='dense' else moe),'--ctx','8192',
                     '--kv-f16','--temp','0','--seed','1','--no-prefix-cache','-n','2048',
                     '--max-tool-rounds','12','--chdir',str(work),'--store-dir',str(run_dir/'store'),
                     '--conversation','pilot','--trace',str(run_dir/'trace.jsonl'),'-p',prompt]
            if mode!='dense': command+=['--nothink']
            else:
                command[command.index('-n')+1]='8192'
                command+=['--hybrid-dense','--expert-think','on','--expert-max-tokens','8192','--expert-max-tool-rounds','12']
            if mode=='hybrid': command+=['--expert-model',str(dense),'--expert-mode','auto']
            (run_dir/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            print(f"START {case['id']} {mode}; cooling {args.cooldown:g}s",flush=True)
            time.sleep(args.cooldown)
            env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_'))}
            env['QW3_METAL_KERNEL_DIR']=str(ROOT/'metal');env['NO_COLOR']='1'
            samples=[];start=time.monotonic(); timed_out=False
            with (run_dir/'output.log').open('w') as out:
                process=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                while process.poll() is None:
                    sample=meter.sample(process.pid);sample['elapsed']=time.monotonic()-start;samples.append(sample)
                    # Dense deadline includes load. Hybrid also allows the principal its own time.
                    limit=1200 if mode=='dense' else 2400
                    if time.monotonic()-start>limit:
                        process.send_signal(2);timed_out=True
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,9);process.wait()
                    time.sleep(1)
            elapsed=time.monotonic()-start
            (run_dir/'memory.json').write_text(json.dumps(samples)+'\n')
            # Run the immutable frozen checker, regardless of model exit/claims.
            authoritative=run_dir/'evaluate.py';authoritative.write_text(case['check'])
            evaluation_env=dict(os.environ,PYTHONPATH=str(work))
            try:
                check=subprocess.run(['python3',str(authoritative)],cwd=work,env=evaluation_env,
                                     capture_output=True,text=True,timeout=15)
                passed=check.returncode==0;check_output=check.stdout+check.stderr
            except subprocess.TimeoutExpired:
                passed=False;check_output='authoritative checker timed out\n'
            unchanged=digest(work/'check.py')==before and (work/'README.md').read_text()==case['spec']+'\n'
            (run_dir/'evaluation.log').write_text(check_output)
            row=dict(case=case['id'],difficulty=case['difficulty'],mode=mode,rc=process.returncode,
                     passed=passed and unchanged,timed_out=timed_out,tests_unchanged=unchanged,elapsed_seconds=elapsed,
                     peak_rss=max((s['rss'] for s in samples),default=0),
                     peak_physical=max((s['physical'] for s in samples),default=0),
                     swap_start=samples[0]['swap'] if samples else None,
                     swap_peak=max((s['swap'] or 0 for s in samples),default=0),
                     swap_end=samples[-1]['swap'] if samples else None,**timing(run_dir/'trace.jsonl'))
            trace=(run_dir/'trace.jsonl').read_text()
            row['delegated']='"role":"dense"' in trace
            # Include primary checkpoint/reload timings from child traces as evidence.
            events=[]
            for path in (run_dir/'store').rglob('*.trace.jsonl'):
                for line in path.read_text().splitlines():
                    try: event=json.loads(line)
                    except json.JSONDecodeError: continue
                    if event.get('event')=='hybrid_checkpoint': events.append(event)
            row['checkpoint_ms']=sum(e.get('elapsed_ms',0) for e in events)
            results['runs'].append(row);save()
            print(f"DONE {case['id']} {mode} pass={row['passed']} elapsed={elapsed:.1f}s delegated={row['delegated']}",flush=True)
    results['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()

if __name__=='__main__': main()
