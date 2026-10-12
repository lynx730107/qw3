"""Independent-run replication; adjacent OFF controls are shared, no IID CI."""
import json
from pathlib import Path
import statistics

base = Path(__file__).resolve().parent
m = json.loads((base / 'native_retry/manifest.json').read_text())
assert m['order'] == [0, 1, 0, 1, 0, 1, 0] and len(m['runs']) == 7
assert all(r['exit_code'] == 0 and len(r['samples']) == 3 for r in m['runs'])
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged'))
means = [statistics.mean(s['ms'] for s in r['samples']) for r in m['runs']]
off = means[::2]
on = means[1::2]
controls = [(means[i-1] + means[i+1]) / 2 for i in (1, 3, 5)]
comparisons = []
for i, control in zip((1, 3, 5), controls):
    candidate = means[i]
    comparisons.append({'block': i, 'candidate_mean_ms': candidate,
                        'adjacent_control_mean_ms': control,
                        'throughput_change_percent': (control / candidate - 1) * 100,
                        'before_change_percent': (means[i-1] / candidate - 1) * 100,
                        'after_change_percent': (means[i+1] / candidate - 1) * 100,
                        'beats_both_neighbors': candidate < min(means[i-1], means[i+1])})
result = {'scope': 'fresh constant-K replica, no discarded samples; nearby controls are shared, no independence-based confidence interval',
          'samples_retained': 21, 'run_means_ms': means,
          'off_spread_percent': (max(off) - min(off)) / statistics.mean(off) * 100,
          'on_spread_percent': (max(on) - min(on)) / statistics.mean(on) * 100,
          'candidate_mean_ms': statistics.mean(on),
          'adjacent_control_mean_ms': statistics.mean(controls),
          'throughput_change_percent': (statistics.mean(controls) / statistics.mean(on) - 1) * 100,
          'comparisons': comparisons}
(base / 'native_retry/analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
