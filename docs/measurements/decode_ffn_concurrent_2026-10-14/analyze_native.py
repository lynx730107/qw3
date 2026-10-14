"""Validate all five raw screening runs and calculate local-control effects."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

archive = Path(__file__).resolve().parent
root = archive.parents[2]
base = archive / 'native'
manifest = json.loads((base / 'manifest.json').read_text())
numerical = json.loads((archive / 'manifest.json').read_text())
assert numerical['all_passed']
assert manifest['completed_utc']
assert manifest['order'] == [0, 1, 0, 1, 0] and len(manifest['runs']) == 5
assert all(manifest['source_sha256'][n]==h for n,h in numerical['source_sha256'].items())
assert len(manifest['source_sha256'])==len(numerical['source_sha256'])+1
assert all(manifest[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

for name, expected in manifest['source_sha256'].items():
    frozen = (base / manifest['source_snapshot_paths'][name]).resolve()
    assert sha(root/name) == sha(frozen) == expected, name
for name, expected in {**manifest['binary_sha256'], **manifest['production_binary_sha256']}.items():
    assert sha(root/name) == expected, name
model = Path(manifest['model'])
assert (model.stat().st_size, model.stat().st_mtime_ns) == (manifest['model_size'], manifest['model_mtime_ns'])
means = []
for index, run in enumerate(manifest['runs']):
    assert run['exit_code'] == 0 and run['variant'] == manifest['order'][index]
    assert 'MTL_DEBUG_LAYER' not in run['environment']
    assert run['environment']['QW3_PRIVATE_FFN_CONCURRENT'] == str(run['variant'])
    log = (base / (run['name']+'.stderr')).read_text()
    samples = [{'repetition': int(n), 'tps': float(t), 'ms': float(ms)} for n,t,ms in
               re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms',log)]
    assert samples == run['samples'] and len(samples) == 3
    assert [s['repetition'] for s in samples] == [1,2,3]
    assert all(math.isfinite(s[k]) and s[k] > 0 for s in samples for k in ('ms','tps'))
    assert run['candidate_calls'] == (3007 if run['variant'] else 0)
    assert f"PASS ffn_concurrent_native variant={run['variant']} calls={run['candidate_calls']} additional_weight_buffer_bytes=0" in log
    rows = list(csv.DictReader(io.StringIO((base/(run['name']+'.stdout')).read_text())))
    assert len(rows) == 1
    row = rows[0]
    assert all(int(row[k]) == v for k,v in {'n_prompt':0,'n_gen':32,'n_depth':128,'reps':3,
                                           'session_bytes':176761680}.items())
    mean = statistics.mean(s['ms'] for s in samples)
    assert abs(mean-float(row['avg_ms'])) <= .006
    means.append(mean)

off = means[::2]
spread = 100 * (max(off)/min(off)-1)
blocks = []
for index in (1,3):
    left,right = means[index-1],means[index+1]
    control = (left+right)/2
    blocks.append({'block':index,
        'reference_interpolated_ms':control,'candidate_ms':means[index],
        'reference_interpolated_tps':32000/control,'candidate_tps':32000/means[index],
        'rate_delta_pct':100*(control/means[index]-1),
        'rate_delta_left_pct':100*(left/means[index]-1),
        'rate_delta_right_pct':100*(right/means[index]-1),
        'beats_both_controls':means[index] < min(left,right)})
analysis = {'block_means_ms':means,'samples_retained':15,'samples_discarded':0,
            'off_max_min_spread_pct':spread,'controls_stable_5pct':spread<=5,'candidate_blocks':blocks,
            'aggregate_rate_delta_pct':100*((statistics.mean(b['reference_interpolated_ms'] for b in blocks))/statistics.mean(b['candidate_ms'] for b in blocks)-1),
            'beats_all_adjacent_controls':all(b['beats_both_controls'] for b in blocks),
            'source_snapshot_binary_model_checks_passed':True,
            'swap_before_after_each_run_equal':all(r['swap_before']==r['swap_after'] for r in manifest['runs']),
            'swap_usage_values':sorted({r[k] for r in manifest['runs'] for k in ('swap_before','swap_after')}),
            'scope':'two screening ON blocks with shared controls; no independent paired interval or adoption evidence'}
(base/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps(analysis,indent=2))
