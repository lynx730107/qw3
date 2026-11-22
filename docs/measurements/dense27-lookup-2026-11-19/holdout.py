"""Choose the shortest passing scalar layer range on C, test once on web."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
root=Path(__file__).resolve().parent
ranges=[(0,0),(0,8),(0,16),(0,32),(0,48),(16,64),(32,64),(48,64)]
valid=[]
for line in (root/'layers.log').read_text().splitlines():
    if not line.startswith('rep='):
        continue
    fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
    if float(fields['max_abs'])<=.005 and float(fields['worst_row_rmse'])<=.001:
        valid.append(ranges[int(fields['rep'])])
assert (root/'layers.log').read_text().count('top1_equal=')==len(ranges)
lo,hi=min(valid+[(0,64)],key=lambda x:x[1]-x[0])
source='web'
if len(sys.argv)>1:
    assert sys.argv[1] in ('web','git')
    source=sys.argv[1]
    lo,hi=0,64
print(f'fallback=[{lo},{hi})',flush=True)
turn=json.loads((root/f'{source}.jsonl').read_text().splitlines()[-1])
before=20
ids=[len(turn['prompt']),before]+turn['prompt']+turn['generated'][:before+8]
with tempfile.TemporaryDirectory() as work:
    dst=Path(work)
    subprocess.run([sys.executable,str(root/'layers.py'),work],check=True)
    p=dst/'ablation.c'
    s=p.read_text().replace('const int lo[]={0,0,0,0,0,16,32,48};',f'const int lo[]={{0,{lo}}};').replace(
        'const int hi[]={0,8,16,32,48,64,64,64};',f'const int hi[]={{0,{hi}}};').replace('rep<8;','rep<2;')
    p.write_text(s)
    binary=str(dst/'probe')
    subprocess.run(['cc','-O3','-mcpu=native','-ffast-math','-std=c99','-I.',str(p),'qw3_metal.o',
                    '-lm','-pthread','-framework','Foundation','-framework','Metal','-o',binary],check=True)
    subprocess.run([binary,'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf'],input=' '.join(map(str,ids)),text=True,check=True)
