"""Time the selected layer range in the greedy prototype if both numeric checks pass."""
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
root=Path(__file__).resolve().parent
names=sys.argv[1:] or ['holdout.log']
s=(root/names[0]).read_text()
lo,hi=map(int,re.search(r'fallback=\[(\d+),(\d+)\)',s).groups())
for name in names:
    data=(root/name).read_text()
    assert f'fallback=[{lo},{hi})' in data
    row=next(l for l in data.splitlines() if l.startswith('rep=1 '))
    fields=dict(re.findall(r'(\w+)=([^ ]+)',row))
    if float(fields['max_abs'])>.005 or float(fields['worst_row_rmse'])>.001:
        print('Candidate failed held-out logits. No timing run.')
        sys.exit(0)
with tempfile.TemporaryDirectory() as work:
    dst=Path(work)
    subprocess.run([sys.executable,str(root/'layers.py'),work],check=True)
    src=(root/'spec.c').read_text().replace('#include "qw3.c"','#include "ablation_core.c"')
    marker='            attempts++;'
    assert src.count(marker)==1
    src=src.replace(marker,marker+f'\n            probe_mask=3; probe_lo={lo}; probe_hi={hi};')
    p=dst/'selected.c'
    p.write_text(src)
    obj=dst/'tail.o'
    subprocess.run(['cc','-O3','-mcpu=native','-ffast-math','-fobjc-arc','-I.','-c',str(root/'tail.m'),'-o',str(obj)],check=True)
    binary=dst/'selected'
    subprocess.run(['cc','-O3','-mcpu=native','-ffast-math','-std=c99','-I.',str(p),str(obj),'-lm','-pthread','-framework','Foundation','-framework','Metal','-o',str(binary)],check=True)
    tokens=json.loads((root/'coding.jsonl').read_text().splitlines()[0])['prompt']
    subprocess.run([str(binary),'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf','64'],input=' '.join(map(str,tokens)),text=True,check=True)
