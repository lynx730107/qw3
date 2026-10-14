"""Check every completed raw screening process, including a live prefix."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--partial', action='store_true')
parser.add_argument('--index', action='store_true')
args = parser.parse_args()
archive = Path(__file__).resolve().parent
root = archive.parents[2]
base = archive / 'native'
m = json.loads((base / 'manifest.json').read_text())
numerical = json.loads((archive / 'manifest.json').read_text())
assert numerical['all_passed']
assert all(m['source_sha256'][n]==h for n,h in numerical['source_sha256'].items())
assert len(m['source_sha256'])==len(numerical['source_sha256'])+1
assert m['order'] == [0, 1, 0, 1, 0]
assert m['cooldown_seconds_before_every_process'] == 60


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


for name, digest in m['source_sha256'].items():
    frozen = (base / m['source_snapshot_paths'][name]).resolve()
    assert sha(root / name) == sha(frozen) == digest, name
    if args.index:
        data = subprocess.run(['git', 'show', ':' + str(frozen.relative_to(root))],
                              cwd=root, capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == digest, name
for name, digest in {**m['binary_sha256'], **m['production_binary_sha256']}.items():
    assert sha(root / name) == digest, name
model = Path(m['model'])
assert (model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns'])
means = []
for index, run in enumerate(m['runs']):
    v = m['order'][index]
    assert run['variant'] == v and run['name'] == f'block_{index}_v{v}'
    assert run['exit_code'] == 0
    assert 'MTL_DEBUG_LAYER' not in run['environment']
    assert run['environment']['QW3_PRIVATE_FFN_CONCURRENT'] == str(v)
    command = ['caffeinate', '-disu', '/tmp/qw3-ffn-concurrent-native-bench', '-m', str(model),
               '--metal', '--ngl', '64', '--llama-style', '--ctx-alloc', '256',
               '-p', '0', '-n', '32', '-d', '128', '-r', '3', '--seed', '1', '-t', '4']
    assert run['command'] == command
    log = (base / (run['name'] + '.stderr')).read_text()
    samples = [{'repetition': int(n), 'tps': float(t), 'ms': float(ms)}
               for n, t, ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms', log)]
    assert samples == run['samples'] and len(samples) == 3
    assert [s['repetition'] for s in samples] == [1, 2, 3]
    assert all(math.isfinite(s[k]) and s[k] > 0 for s in samples for k in ('tps', 'ms'))
    assert run['candidate_calls'] == (3007 if v else 0)
    assert f'PASS ffn_concurrent_native variant={v} calls={run["candidate_calls"]} additional_weight_buffer_bytes=0' in log
    rows = list(csv.DictReader(io.StringIO((base / (run['name'] + '.stdout')).read_text())))
    assert len(rows) == 1
    assert all(int(rows[0][k]) == n for k, n in {'n_prompt': 0, 'n_gen': 32,
               'n_depth': 128, 'reps': 3, 'session_bytes': 176761680}.items())
    mean = statistics.mean(s['ms'] for s in samples)
    assert abs(mean - float(rows[0]['avg_ms'])) <= .006
    means.append(mean)
result = {'index_verified': args.index, 'complete': bool(m.get('completed_utc')),
          'source_references_verified': len(m['source_sha256']),
          'validated_runs': len(means), 'validated_samples': 3 * len(means),
          'sources_snapshots_binaries_model_match': True,
          'scope': 'screening only; no independent paired confidence interval or fresh llama comparison'}
if not args.partial:
    assert len(means) == 5 and m['completed_utc']
    assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged'))
    a = json.loads((base / 'analysis.json').read_text())
    assert a['block_means_ms'] == means and a['samples_retained'] == 15
    assert a['samples_discarded'] == 0
    spread = 100 * (max(means[::2]) / min(means[::2]) - 1)
    assert math.isclose(a['off_max_min_spread_pct'], spread, abs_tol=1e-9)
    for index in (1, 3):
        item = a['candidate_blocks'][0 if index==1 else 1]
        assert item['block']==index
        left, right = means[index - 1], means[index + 1]
        assert math.isclose(item['rate_delta_pct'], 100 * ((left + right) / 2 / means[index] - 1), abs_tol=1e-9)
        assert math.isclose(item['rate_delta_left_pct'], 100 * (left / means[index] - 1), abs_tol=1e-9)
        assert math.isclose(item['rate_delta_right_pct'], 100 * (right / means[index] - 1), abs_tol=1e-9)
        assert item['beats_both_controls'] == (means[index] < min(left, right))
    controls=statistics.mean(a['candidate_blocks'][j]['reference_interpolated_ms'] for j in (0,1))
    on=statistics.mean(means[j] for j in (1,3))
    assert math.isclose(a['aggregate_rate_delta_pct'],100*(controls/on-1),abs_tol=1e-9)
    result.update(analysis=a, swap_before_after_each_run_equal=all(
                  r['swap_before'] == r['swap_after'] for r in m['runs']))
    (base / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
