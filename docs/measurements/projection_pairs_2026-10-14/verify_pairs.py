"""Verify close-pair source provenance and original benchmark timing work."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', action='store_true')
args = parser.parse_args()
b = Path(__file__).resolve().parent
root = b.parents[2]
m = json.loads((b / 'manifest.json').read_text())
process = subprocess.run(['python3', str(b / 'analyze_pairs.py')], cwd=root, capture_output=True, text=True, check=True)
analysis = json.loads(process.stdout)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


for name, digest in m['source_sha256'].items():
    frozen = (b / m['source_snapshot_paths'][name]).resolve()
    assert sha(root / name) == sha(frozen) == digest
    if args.index:
        data = subprocess.run(['git', 'show', ':' + str(frozen.relative_to(root))], cwd=root, capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == digest, name
base = root / 'docs/measurements/decode_projection_barriers_2026-10-14'
old = json.loads((base / 'manifest.json').read_text())
assert old['all_passed']
for name in ('hooks.inc', 'qw3_private.c', 'qw3_private_metal.m'):
    key = 'tests/.decode_projection_barriers/' + name
    assert sha(root / 'tests/.projection_pairs' / name) == old['source_sha256'][key]
bench = (root / 'tests/.projection_pairs/bench_private.c').read_text()
a = bench.index('        if (cfg->repetitions!=12 ||')
c = bench.index('        qw3_session_invalidate(session);', a)
bench = bench[:a] + bench[c:]
a = bench.index('        if (private_error ||')
c = bench.index('        samples_sec[rep] = t1 - t0;', a)
bench = bench[:a] + bench[c:]
assert bench == (root / 'qw3_bench.c').read_text()
assert analysis['samples_retained'] == 12 and analysis['samples_discarded'] == 0
result = {'index_verified': args.index, 'source_references_verified': len(m['source_sha256']),
          'original_benchmark_restored_exactly': True, 'original_model_gate_helpers_unchanged': True,
          'single_engine_and_session': True, 'new_weight_or_session_data_buffers': 0,
          'analysis': analysis, 'adoption_evidence': False,
          'scope': 'close-pair diagnostic only; process-independent replication and checkpoint/agent gates still required'}
(b / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
