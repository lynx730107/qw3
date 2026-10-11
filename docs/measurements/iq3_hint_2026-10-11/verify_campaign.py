"""Validate all raw timing samples, metadata and immutable source references."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics

screen = Path(__file__).resolve().parent
root = screen.parents[2]
for name in ['native', 'native_retry']:
    p = screen / name
    m = json.loads((p / 'manifest.json').read_text())
    assert len(m['runs']) == 5
    assert m['source_unchanged'] and m['binary_unchanged'] and m['model_unchanged']
    means = []
    for r in m['runs']:
        log = (p / (r['name'] + '.stderr')).read_text()
        samples = [{'repetition': int(n), 'tps': float(tps), 'ms': float(ms)}
                   for n, tps, ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms', log)]
        assert r['exit_code'] == 0 and samples == r['samples'] and len(samples) == 3
        calls = re.findall(r'PASS iq3_hint_native variant=(\d+) calls=(\d+) additional_weight_buffer_bytes=0', log)
        assert calls == [(str(r['variant']), str(r['candidate_calls']))]
        assert r['candidate_calls'] == (7081 if r['variant'] else 0)
        meta = re.findall(r'iq3_hint_metadata variant=(\d+) thread_width=(\d+) max_threads=(\d+) static_threadgroup_bytes=(\d+)', log)
        assert meta == [('0', '32', '1024', '1024'), ('1', '32', '64', '1024'), ('2', '32', '64', '1024')]
        row = list(csv.DictReader((p / (r['name'] + '.stdout')).read_text().splitlines()))
        mean = statistics.mean(s['ms'] for s in samples)
        means.append(mean)
        assert len(row) == 1 and row[0]['test'] == 'tg'
        assert row[0]['n_gen'] == '32' and row[0]['n_depth'] == '128' and row[0]['reps'] == '3'
        assert abs(float(row[0]['avg_ms']) - mean) <= .00501
        assert int(row[0]['session_bytes']) == 176761680
    for key in ['binary_sha256', 'production_binary_sha256']:
        for path, digest in m[key].items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    for path, digest in m['source_sha256'].items():
        assert hashlib.sha256((p / m['source_snapshot_paths'][path]).resolve().read_bytes()).hexdigest() == digest
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    st = Path(m['model']).stat()
    assert (st.st_size, st.st_mtime_ns) == (m['model_size'], m['model_mtime_ns'])
    off = [means[i] for i in [0, 2, 4]]
    spread = (max(off) - min(off)) / statistics.mean(off) * 100
    comparisons = []
    for i in [1, 3]:
        ref = (means[i - 1] + means[i + 1]) / 2
        comparisons.append({'block': i, 'variant': m['runs'][i]['variant'], 'candidate_mean_ms': means[i],
                            'adjacent_off_mean_ms': ref, 'point_gain_pct': (ref / means[i] - 1) * 100,
                            'gain_vs_left_off_pct': (means[i - 1] / means[i] - 1) * 100,
                            'gain_vs_right_off_pct': (means[i + 1] / means[i] - 1) * 100,
                            'neighbor_interpolated_saved_ms': ref - means[i]})
    a = {'adopted': False, 'runs': 5, 'samples_retained': 15, 'samples_discarded': 0,
         'block_means_ms': means, 'off_mean_spread_pct': spread, 'control_stability_gate_passed': spread <= 5,
         'additional_weight_buffer_bytes': 0, 'additional_session_data_buffers': 0,
         'native_session_bytes': 176761680, 'comparisons': comparisons,
         'swap_observations': [{k: r[k] for k in ['name', 'swap_before', 'swap_after']} for r in m['runs']]}
    if name == 'native_retry':
        refs = statistics.mean(c['adjacent_off_mean_ms'] for c in comparisons)
        candidate = statistics.mean(means[i] for i in [1, 3])
        a['neighbor_weighted_control_mean_ms'] = refs
        a['candidate_mean_ms'] = candidate
        a['gain_vs_neighbor_weighted_controls_pct'] = (refs / candidate - 1) * 100
    else:
        a['reason'] = 'OFF mean spread exceeds 5%; asymmetric apparent SIMD gain is inconclusive. Retain all samples and replicate mode 2.'
    (p / 'analysis.json').write_text(json.dumps(a, indent=2) + '\n')
    print(json.dumps({'campaign': name, **a}, indent=2))
