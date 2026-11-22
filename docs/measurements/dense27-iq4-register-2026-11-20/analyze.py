"""Compare 32-token decode latencies within each balanced three-way run."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
names = {
    'paired': ['IQ4 originale', 'IQ4 lookup packed', 'IQ4 simd_shuffle'],
    'paired3': ['IQ3 originale', 'IQ3 scala finale', 'IQ3 dot vettoriale'],
    'grid': ['IQ3 originale', 'IQ3 tabella precalcolata condivisa', 'IQ3 lettura diretta costante'],
}
all_runs = {}
for log, labels in names.items():
    path = root / f'{log}.log'
    if not path.exists():
        continue
    rows = {}
    quality = {}
    for line in path.read_text().splitlines():
        fields = dict(re.findall(r'(\w+)=([^ ]+)',line))
        if line.startswith('quality '):
            quality[int(fields['variant'])] = fields
        if line.startswith('time '):
            rows[int(fields['block']),int(fields['variant'])] = float(fields['ms'])
    if len(rows) != 18:
        continue
    clocks = [[rows[b,v] for b in range(6)] for v in range(3)]
    runs = {}
    for v,label in enumerate(labels):
        tps = 192000/sum(clocks[v])
        runs[label] = dict(tps=tps, ms_per_token=1000/tps,
                           delta_percent=100*(sum(clocks[0])/sum(clocks[v])-1),
                           faster_blocks=sum(x < y for x,y in zip(clocks[v],clocks[0])),
                           logits=quality[v])
    all_runs[log] = runs
print(json.dumps(all_runs,indent=2))
