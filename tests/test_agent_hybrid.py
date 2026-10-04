#!/usr/bin/env python3
"""Real MoE -> dense -> MoE safety checks. Runs models strictly sequentially.

Requires both GGUFs and Metal. Artifacts are retained under --output. All writes
and interruption fixtures are isolated; no rollback or automatic tool replay.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
MOE=(ROOT/'../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf').resolve()
DENSE=(ROOT/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()

def events(path):
    if not path.exists(): return []
    result=[]
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass
    return result

def assert_serial(rows):
    live=None
    for row in rows:
        if row['event']=='hybrid_process_start':
            assert live is None,('overlapping models',live,row)
            live=row['pid']
        elif row['event']=='hybrid_process_exit':
            assert live==row['pid'],row
            live=None
    assert live is None,('unreaped child',live)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cases',default='default_thinking,return_large_audit,token_budget,edit,timeout,missing,interrupt_generation,interrupt_tool,crash,duplicate,incompatible')
    args=ap.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_'))}
    env['QW3_METAL_KERNEL_DIR']=str(ROOT/'metal');env['NO_COLOR']='1'
    summary=[]
    def launch(name,prompt,extra=(),store=None):
        d=args.output/name; d.mkdir(exist_ok=True);work=d/'work';work.mkdir(exist_ok=True)
        (work/'README.md').write_text('Isolated hybrid safety fixture.\n')
        if name=='return_large_audit':
            (work/'evidence.txt').write_text(('river mountain ocean valley forest meadow island desert plain lake plateau\n')*180)
        command=[str(ROOT/'qw3-agent'),'--metal','-m',str(MOE),'--expert-model',str(DENSE),
                 '--expert-mode','manual','--nothink','--expert-think','none','--ctx','8192',
                 '--kv-f16','--temp','0','--seed','1','-n','384','--no-prefix-cache',
                 '--chdir',str(work),'--store-dir',str(store or d/'store'),'--conversation','safety',
                 '--trace',str(d/'trace.jsonl'),'-p',prompt,*extra]
        if name=='default_thinking':
            index=command.index('--expert-think'); del command[index:index+2]
        (d/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        log=(d/'output.log').open('w')
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return d,work,p,log
    def wait(p,log):
        try: p.wait(timeout=360)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise AssertionError('safety fixture timed out')
        finally: log.close()
    def until(condition,p,seconds=180):
        deadline=time.monotonic()+seconds
        while not condition():
            assert p.poll() is None,('worker exited before trigger',p.returncode)
            assert time.monotonic()<deadline,'trigger timed out'
            time.sleep(.1)
    for name in args.cases.split(','):
        print('START safety',name,flush=True)
        if name=='default_thinking':
            d,w,p,log=launch(name,'/expert run Leggi README.md con read. Rispondi THINK_DEFAULT_OK senza modificare file.')
            wait(p,log); assert p.returncode==0
            output=(d/'output.log').read_text()
            assert 'invalid expert thinking mode' not in output and 'THINK_DEFAULT_OK' in output
            assert '--expert-think' in output or 'think=think' in output
            assert_serial(events(d/'trace.jsonl'))
        elif name=='return_large_audit':
            d,w,p,log=launch(name,'/expert run Leggi evidence.txt con read, lines=180. Rispondi soltanto LARGE_AUDIT_OK. Non modificare file.')
            wait(p,log); assert p.returncode==0
            job=next((d/'store'/'hybrid').glob('job-*'))
            result=(job/'result.txt').read_text()
            assert len(result)>12000 and result.index('Expert answer:')<1000,result[:1000]
            assert 'LARGE_AUDIT_OK' in result[:12000]
            assert_serial(events(d/'trace.jsonl'))
        elif name=='token_budget':
            d,w,p,log=launch(name,'/expert run Leggi README.md con read e descrivilo in almeno 100 parole.',
                           ['--expert-max-tokens','1'])
            wait(p,log); assert p.returncode==0
            result=next((d/'store').rglob('result.txt')).read_text()
            assert 'Expert status: failed/incomplete' in result,result
            assert 'Generated tokens used: 1/1' in result,result
            assert 'expert total token budget exhausted' in (d/'output.log').read_text()
            assert_serial(events(d/'trace.jsonl'))
        elif name=='edit':
            d,w,p,log=launch(name,"/expert run Usa il tool write per creare answer.txt con contenuto HYBRID_EDIT_OK. Poi usa bash per eseguire: python3 -c \"from pathlib import Path; assert Path('answer.txt').read_text().strip() == 'HYBRID_EDIT_OK'; print('VERIFY_OK')\". Rispondi soltanto con l'esito verificato.")
            wait(p,log);assert p.returncode==0
            assert (w/'answer.txt').read_text().strip()=='HYBRID_EDIT_OK'
            audit=list((d/'store').rglob('dense-tools.jsonl'));assert len(audit)==1
            rows=events(audit[0]);assert any(r['stage']=='started' and r['tool']=='write' for r in rows)
            assert any(r['stage']=='completed' and r['tool']=='bash' and 'VERIFY_OK' in r['output'] for r in rows)
            assert_serial(events(d/'trace.jsonl'))
        elif name in ('timeout','missing','interrupt_generation'):
            extra=['--expert-timeout','1'] if name=='timeout' else ['--expert-model',str(args.output/'absent.gguf')] if name=='missing' else []
            d,w,p,log=launch(name,'/expert run Scrivi un ragionamento tecnico dettagliato di almeno 500 parole sul protocollo di recupero dopo un crash. Non modificare file.',extra)
            if name=='interrupt_generation':
                until(lambda:any(e['event']=='hybrid_ready' and e.get('role')=='dense' for e in events(d/'trace.jsonl')),p)
                time.sleep(3);p.send_signal(signal.SIGINT)
            wait(p,log);assert p.returncode==0,(d/'output.log').read_text()[-2000:]
            assert_serial(events(d/'trace.jsonl'))
            assert len(list((d/'store').rglob('result.txt')))==1
            assert not list((d/'store'/'hybrid').glob('active-*'))
        elif name in ('interrupt_tool','crash'):
            task='/expert run Usa subito bash per eseguire esattamente: python3 -c "from pathlib import Path; import time; p=Path(\'audit.txt\'); p.write_text(p.read_text()+\'one\\n\' if p.exists() else \'one\\n\'); time.sleep(60)" . Poi riferisci il risultato. Non usare altri tool prima di bash.'
            d,w,p,log=launch(name,task)
            until(lambda:(w/'audit.txt').exists(),p)
            p.send_signal(signal.SIGKILL if name=='crash' else signal.SIGINT);wait(p,log)
            assert (w/'audit.txt').read_text()=='one\n'
            if name=='interrupt_tool':
                assert p.returncode==0
                result=next((d/'store').rglob('result.txt')).read_text()
                assert 'Verification pending: yes' in result,result
                assert_serial(events(d/'trace.jsonl'))
            else:
                # IPC EOF lets the orphan stop its shell and release its model lock.
                time.sleep(6)
                restart=args.output/'crash-restart';restart.mkdir()
                command=json.loads((d/'command.json').read_text());command[command.index('-p')+1]='Non ripetere il comando originale. Verifica usando solo bash: python3 -c "from pathlib import Path; assert Path(\'audit.txt\').read_text() == \'one\\n\'; print(\'RECOVERY_OK\')". Poi riferisci la verifica.'
                command[command.index('--trace')+1]=str(restart/'trace.jsonl')
                with (restart/'output.log').open('w') as out:
                    r=subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=180)
                assert r.returncode==0
                assert 'saved tools will not be replayed' in (restart/'output.log').read_text()
                assert (w/'audit.txt').read_text()=='one\n','mutating tool replayed'
                assert not any(e.get('role')=='dense' for e in events(restart/'trace.jsonl'))
                assert_serial(events(restart/'trace.jsonl'))
        elif name in ('duplicate','incompatible'):
            # Seed a successful handoff, then emulate an unacknowledged return.
            d,w,p,log=launch(name,'/expert run Leggi README.md con read e rispondi SEED_OK. Non modificare file.')
            wait(p,log);assert p.returncode==0
            job=next((d/'store'/'hybrid').glob('job-*'));(job/'phase').write_text('3\n')
            primary=next((d/'store'/'hybrid').glob('moe-*'))
            if name=='incompatible':
                (primary/'safety.kvc').write_bytes(b'corrupt checkpoint')
                (primary/'safety.hybrid-state').write_bytes(b'corrupt runtime')
            rows=events(d/'trace.jsonl');prof=1469598103934665603
            # Recover active name from stable parent model path + int64 size + session.
            for data in [str(MOE).encode(),MOE.stat().st_size.to_bytes(8,'little',signed=True),b'safety']:
                for byte in data: prof=((prof^byte)*1099511628211)&((1<<64)-1)
            (d/'store'/'hybrid'/f'active-{prof:016x}').write_text(str(job))
            command=json.loads((d/'command.json').read_text());command[command.index('-p')+1]='Rispondi soltanto RESUME_OK.'
            command[command.index('--trace')+1]=str(d/'restart.trace.jsonl')
            with (d/'restart.log').open('w') as out:
                r=subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=180)
            assert r.returncode==0
            text=(primary/'safety.txt').read_text();assert text.count('Hybrid result ID: ')==1,text
            assert not any(e.get('role')=='dense' for e in events(d/'restart.trace.jsonl'))
            assert_serial(events(d/'restart.trace.jsonl'))
        summary.append(dict(case=name,passed=True))
        (args.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('PASS safety',name,flush=True)

if __name__=='__main__':main()
