#!/usr/bin/env python3
"""Validate raw balanced public-policy reference runs and recompute statistics."""
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
parser.add_argument('--partial', action='store_true', help='validate the completed prefix without writing final analysis')
parser.add_argument('--index', action='store_true', help='also verify staged source snapshots')
parser.add_argument('--results-file', type=Path, default=None,
                    help='validate an immutable intermediate snapshot against the same raw logs')
parser.add_argument('--envelope-file', type=Path, default=None)
args = parser.parse_args()
base = Path(__file__).resolve().parent
root = base.parents[2]
archive = base/'public_reference/runs'
data = json.loads((args.results_file or (archive / 'results.json')).read_text())
envelope = json.loads((args.envelope_file or (archive.parent/'envelope.json')).read_text())
integration = json.loads((base/'public_integration_v2/manifest.json').read_text())
worktree = Path(integration['worktree'])

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert 'failure' not in data
assert data['qw3_variant'] == 'ordinary'
assert data['pairs'] == 6 and data['repetitions'] == 3
assert data['order_balanced'] and data['cooldown_scope'] == 'run'
assert data['cooldown_seconds_before_engine_run'] == 60
assert data['context'] == 256 and data['kv'] == 'f16'
assert data['full_trunk_and_head_offload'] and not data['mtp']
assert not data['same_token_stream'] and data['warmup']
assert data['environment_prefixes_cleared'] == ['QW3_', 'GGML_', 'MTL_']
assert envelope['dyld_overrides_absent']
assert set(data['source_sha256']).issubset(envelope['source_snapshot_paths'])
for name, expected in data['source_sha256'].items():
    snapshot = (archive.parent / envelope['source_snapshot_paths'][name]).resolve()
    assert sha(worktree / name) == sha(snapshot) == expected, name
    if args.index:
        staged = subprocess.run(['git', '-C', str(root), 'show', ':' + str(snapshot.relative_to(root))],
                                capture_output=True, check=True).stdout
        assert hashlib.sha256(staged).hexdigest() == expected, name
for name, expected in data['binary_sha256'].items():
    assert sha(name) == expected, name
libraries = envelope['llama_library_sha256']
assert len(libraries) >= 8
for name, expected in libraries.items():
    assert sha(name) == expected, name
model = Path(data['model'])
assert model.stat().st_size == data['model_size']
assert model.stat().st_mtime_ns == data['model_mtime_ns']

jobs = {'pp128': (128, 0, 0), 'tg32_d128': (0, 32, 128)}
order = [(job, pair, engine) for job in jobs for pair in range(6)
         for engine in (['qw3', 'llama'] if pair % 2 == 0 else ['llama', 'qw3'])]
observed = [(r['job'], r['pair'], r['engine']) for r in data['runs']]
assert observed == order[:len(observed)]
if not args.partial:
    assert observed == order and data['completed_utc']
    assert data['sources_unchanged'] and data['binaries_unchanged']
    assert data['samples_discarded'] == 0

timings = {job: {'qw3': [], 'llama': []} for job in jobs}
session_bytes = {job: set() for job in jobs}
swap = []
sample_count = 0
for run in data['runs']:
    job, engine = run['job'], run['engine']
    pp, tg, depth = jobs[job]
    assert run['environment'] == {'QW3_METAL_KV_F16': '1', 'QW3_METAL_KV_Q8_0': '0',
                                  'QW3_BENCH_PRINT_SAMPLES': '1'}
    command = run['command']
    assert command[:2] == ['caffeinate', '-disu']
    expected_binary = next(p for p in data['binary_sha256'] if Path(p).name ==
                           ('qw3-bench' if engine == 'qw3' else 'llama-bench'))
    assert command[2] == expected_binary
    common = ['-m', str(model)]
    if engine == 'qw3':
        expected_args = common + ['--metal', '--ngl', '64', '--llama-style', '--ctx-alloc', '256',
                                 '-p', str(pp), '-n', str(tg), '-d', str(depth), '-r', '3',
                                 '--seed', '1', '-t', '4']
    else:
        expected_args = common + ['-ngl', '999', '-ctk', 'f16', '-ctv', 'f16', '-fa', 'on',
                                 '-b', '128', '-ub', '128', '-p', str(pp), '-n', str(tg),
                                 '-d', str(depth), '-r', '3', '-t', '4', '-o', 'json', '-v']
    assert command[3:] == expected_args
    stem = archive / f"{job}-{run['pair']}-{engine}"
    stdout = stem.with_suffix('.stdout').read_text()
    stderr = stem.with_suffix('.stderr').read_text()
    samples = run['samples']
    assert len(samples) == 3
    assert [s['repetition'] for s in samples] == list(range(1, 4))
    assert all(math.isfinite(s[k]) and s[k] > 0 for s in samples for k in ('ms', 'tok_s'))
    if engine == 'qw3':
        rows = list(csv.DictReader(io.StringIO(stdout)))
        assert rows == [run['raw']]
        row = rows[0]
        session_bytes[job].add(int(row['session_bytes']))
        assert all(int(row[k]) == v for k, v in
                   {'n_prompt': pp, 'n_gen': tg, 'n_depth': depth, 'reps': 3}.items())
        parsed = re.findall(r'qw3-bench: sample (\d+)/(\d+) ([\d.]+) tok/s ([\d.]+) ms', stderr)
        assert len(parsed) == 3
        for s, (i, total, tps, ms) in zip(samples, parsed):
            assert int(i) == s['repetition'] and int(total) == 3
            assert s['tok_s'] == float(tps) and s['ms'] == float(ms)
        assert abs(statistics.mean(s['ms'] for s in samples) - float(row['avg_ms'])) <= .006
        seconds = float(row['avg_ms']) / 1000
    else:
        rows = json.loads(stdout)
        assert rows == [run['raw']]
        row = rows[0]
        assert (row['n_prompt'], row['n_gen'], row['n_depth']) == (pp, tg, depth)
        assert row['type_k'] == row['type_v'] == 'f16'
        assert row['n_gpu_layers'] == 999 and row['flash_attn'] == 1
        assert row['n_batch'] == row['n_ubatch'] == 128 and row['n_threads'] == 4
        assert {k: row[k] for k in ('build_commit', 'build_number')} == data['llama_binary_build']
        offload = re.search(r'offloaded\s+(\d+)/(\d+)\s+layers', stderr)
        assert offload and offload[1] == offload[2]
        assert 'offloading output layer to GPU' in stderr
        assert re.search(r'n_ctx\s*=\s*256\b', stderr)
        assert len(row['samples_ns']) == len(row['samples_ts']) == 3
        for s, ns, tps in zip(samples, row['samples_ns'], row['samples_ts']):
            assert s['ms'] == ns / 1e6 and s['tok_s'] == tps
        assert abs(statistics.mean(row['samples_ns']) - row['avg_ns']) <= 1
        seconds = row['avg_ns'] / 1e9
    assert run['seconds'] == seconds
    timings[job][engine].append(seconds)
    swap.extend([run['swap_before'], run['swap_after']])
    sample_count += len(samples)

analyses = {}
for job, summary in data['summary'].items():
    t = timings[job]
    assert len(t['qw3']) == len(t['llama']) == 6
    differences = [(a - b) * 1000 for a, b in zip(t['llama'], t['qw3'])]
    mean = statistics.mean(differences)
    radius = 2.571 * statistics.stdev(differences) / math.sqrt(6)
    tokens = sum(jobs[job][:2])
    rates = {engine: tokens / statistics.mean(v) for engine, v in t.items()}
    spreads = {engine: 100 * (max(v) / min(v) - 1) for engine, v in t.items()}
    expected = {'qw3_tps': rates['qw3'], 'llama_tps': rates['llama'],
                'paired_saved_ms': mean, 'ci95_radius_ms': radius,
                'qw3_spread_pct': spreads['qw3'], 'llama_spread_pct': spreads['llama'],
                'saved_ms_qw3_first_mean': statistics.mean(differences[::2]),
                'saved_ms_qw3_second_mean': statistics.mean(differences[1::2])}
    for key, value in expected.items():
        assert math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-9), (job, key)
    stable = all(s <= 5 for s in spreads.values())
    assert summary['both_engines_stable_5pct'] == stable
    assert summary['lead_confirmed_in_this_series'] == (mean - radius > 0)
    assert len(session_bytes[job]) == 1
    analyses[job] = {'paired_saved_ms': differences, 'paired_saved_ci95_ms': [mean-radius, mean+radius],
                     'rate_delta_pct': 100 * (rates['qw3'] / rates['llama'] - 1),
                     'qw3_speedup_needed_pct': 100 * (rates['llama'] / rates['qw3'] - 1),
                     'qw3_lead_confirmed_in_this_series': stable and mean-radius > 0,
                     'llama_lead_confirmed_in_this_series': stable and mean+radius < 0,
                     'both_engines_stable_5pct': stable,
                     'qw3_session_bytes': next(iter(session_bytes[job]))}

analysis = {'complete': bool(data.get('completed_utc')), 'validated_runs': len(data['runs']),
            'validated_samples': sample_count, 'all_sources_snapshots_binaries_libraries_match': True,
            'jobs': analyses, 'swap_usage_values': sorted(set(swap)),
            'swap_before_after_each_run_equal': all(r['swap_before'] == r['swap_after'] for r in data['runs']),
            'interval_experimental_units': 'six paired engine-run means per job; three repetitions each',
            'interval_scope': 'conditional on independence of temporally separated paired means',
            'streams': 'native synthetic token streams differ between engines; no sampling or MTP',
            'memory_scope': 'session_bytes is reported by qw3; process peak RAM was not measured'}
if not args.partial:
    assert envelope['completed_utc'] and envelope['exit_code']==0
    assert all(envelope[k] for k in ('public_source_unchanged','public_binary_unchanged','llama_library_unchanged','benchmark_driver_unchanged'))
    assert set(analyses) == set(jobs) and sample_count == 72
    analysis['samples_discarded'] = data['samples_discarded']
    (archive / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
print(json.dumps(analysis, indent=2))
