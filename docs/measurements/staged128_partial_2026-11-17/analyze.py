"""Summarize all retained samples, compare adjacent processes when balanced."""
from pathlib import Path
from statistics import mean, stdev
import json
import re

root = Path(__file__).resolve().parent
report = {}
for name in ('baseline', 'screen', 'formats', 'replica'):
    data = json.loads((root / f'{name}.json').read_text())
    if name == 'baseline':
        runs = [dict(n=r['tokens'], mode=0, ms=r['ms']) for r in data]
    else:
        assert data['complete']
        runs = data['runs']
    by_length = {}
    for length in sorted({r['n'] for r in runs}):
        blocks = [r for r in runs if r['n'] == length]
        modes = sorted({r['mode'] for r in blocks})
        stats = {}
        for mode in modes:
            matching = [r for r in blocks if r['mode'] == mode]
            samples = [x for r in matching for x in r['ms']]
            mu = mean(samples)
            stats[mode] = {'samples': len(samples), 'mean_ms': mu, 'sd_ms': stdev(samples),
                           'tokens_per_s': length * 1000 / mu, 'process_means_ms': [mean(r['ms']) for r in matching]}
        full = {'statistics': stats}
        if len(modes) == 2:
            paired = []
            for i in range(0, len(blocks), 2):
                ref, test = sorted(blocks[i:i+2], key=lambda r: r['mode'])
                assert (ref['mode'], test['mode']) == tuple(modes)
                paired.append(mean(ref['ms']) - mean(test['ms']))
            ref, test = [stats[m]['mean_ms'] for m in modes]
            full.update(paired_ms_saved=paired, latency_reduction_pct=100*(1-test/ref),
                        throughput_gain_pct=100*(ref/test-1))
            critical = {2: 12.7062, 4: 3.1824}[len(paired)]
            half_width = critical*stdev(paired)/(len(paired)**.5)
            full['paired_t95_ms_saved'] = [mean(paired)-half_width, mean(paired)+half_width]
        by_length[length] = full
    report[name] = by_length
payload_bytes = {}
for series in ('replica', 'screen', 'formats'):
    for log in root.glob(f'{series}-[0-9]*-v[0-9].log'):
        match = re.search(r'^pp,(\d+),0,0,3,[\d.,]+,(\d+)$', log.read_text(), re.M)
        assert match, log.name
        payload_bytes.setdefault(match[1], set()).add(int(match[2]))
report['payload_bytes_by_length'] = {key: sorted(value) for key, value in payload_bytes.items()}
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for length, info in report['replica'].items():
    print(length, info)
