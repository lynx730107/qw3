"""Nearby-control screening: retain every ordinary sample and order effect."""
import json
from pathlib import Path
import statistics

base = Path(__file__).resolve().parent
m = json.loads((base / 'native/manifest.json').read_text())
assert len(m['runs']) == 5
assert all(r['exit_code'] == 0 and len(r['samples']) == 3 for r in m['runs'])
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged'))
means = [statistics.mean(s['ms'] for s in r['samples']) for r in m['runs']]
off = [means[i] for i in (0, 2, 4)]
spread = (max(off) - min(off)) / statistics.mean(off) * 100
comparisons = []
for i in (1, 3):
    before, candidate, after = means[i-1:i+2]
    control = (before + after) / 2
    comparisons.append({
        'variant': m['runs'][i]['variant'], 'candidate_mean_ms': candidate,
        'adjacent_control_mean_ms': control,
        'throughput_change_percent': (control / candidate - 1) * 100,
        'before_change_percent': (before / candidate - 1) * 100,
        'after_change_percent': (after / candidate - 1) * 100,
        'beats_both_neighbors': candidate < min(before, after),
    })
result = {'scope': 'initial five-run screening, not independent-replica evidence or a llama comparison',
          'samples_retained': sum(len(r['samples']) for r in m['runs']),
          'run_means_ms': means, 'off_spread_percent': spread,
          'off_spread_within_5_percent': spread <= 5,
          'comparisons': comparisons,
          'replication_required_before_adoption': True}
(base / 'native/analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
