"""Run same-session replicas only for candidates that passed both model and native screens."""
from pathlib import Path
import json,os,subprocess,sys
out=Path(__file__).resolve().parent
root=out.parents[2];work=Path(sys.argv[1]).resolve()
for kind,probe,variant,calls in [('gqa','gqa_heads_probe','gqa_groups',16),('nibble','iq3_nibble','iq3_nibble',73)]:
    if sys.argv[2]=='build':
        directory=work/'tests'/('.'+probe)
        text='#include "hooks.inc"\n'
        for name,value in [('COMPILE',variant+'_compile'),('VARIANT','groups_variant' if kind=='gqa' else 'nibble_variant'),('CALLS',variant+'_calls'),('LAYER_CALLS',str(calls))]:
            text+=f'#define CANDIDATE_{name} {value}\n'
        (directory/'paired.m').write_text(text+(out/'paired.inc').read_text())
        subprocess.run(['cc','-O3','-ffast-math','-mcpu=native','-fno-finite-math-only','-Wall','-Wextra','-fobjc-arc','-I.','-Itests',str(directory/'paired.m'),'-lm','-pthread','-framework','Foundation','-framework','Metal','-o',kind+'_paired'],cwd=work,check=True)
    else:
        screens={r['name']:r['exit'] for r in json.loads((out/'screen.json').read_text())}
        if screens.get(kind+'-model')!=0 or screens.get(kind+'-tg-7-v1')!=0: continue
        env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','MTL_','GGML_'))}
        env.update(QW3_METAL_KV_F16='1',QW3_METAL_KV_Q8_0='0',QW3_METAL_NGL='64')
        model=(root/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
        with (out/(kind+'-paired.log')).open('x') as log:
            subprocess.run(['caffeinate','-disu',str(work/(kind+'_paired')),str(model)],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
