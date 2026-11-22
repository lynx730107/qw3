"""Print end-to-end throughput from the mean latency of all retained samples."""
from pathlib import Path
from statistics import mean, stdev
import json
import sys

archive = Path(__file__).resolve().parent
series = sys.argv[1] if len(sys.argv) > 1 else 'screen'
run = json.loads((archive/f'{series}.json').read_text())
assert run['completed']
groups = {}
for sample in run['runs']:
    groups.setdefault(sample['mode'], []).extend(sample['ms'])
base = mean(groups[0])
results = []
for mode, times in sorted(groups.items()):
    latency = mean(times)
    nearby = [p['throughput_gain_pct'] for p in run['comparisons'] if p['mode'] == mode]
    results.append(dict(mode=mode, count=len(times), mean_ms=latency, sd_ms=stdev(times),
                        tok_s=32000/latency, gain_pct=100*(base/latency-1), nearby_pct=nearby))
off_means = [p['mean_ms'] for p in run['runs'] if p['mode']==0]
analysis = {'results':results,'baseline_spread_pct':100*(max(off_means)/min(off_means)-1),
            'raw_samples':sum(len(v) for v in groups.values())}
print(json.dumps(analysis,indent=2))
(archive/f'{series}-analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
