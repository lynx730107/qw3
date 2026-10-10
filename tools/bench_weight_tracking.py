#!/usr/bin/env python3
"""Archive two private paired screenings; not a native throughput comparison."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cooldown', type=float, default=60)
args = parser.parse_args()
if not 0 <= args.cooldown <= 60:
    parser.error('cooldown must be 0..60')
binary = args.binary.resolve(strict=True)
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
files = [root / n for n in ('Makefile', 'qw3.c', 'qw3.h', 'qw3_metal.m', 'qw3_metal.h',
    'qw3_bench.c', 'qw3_runtime_fingerprint.h', 'tests/weight_tracking_candidates.inc',
    'tests/prepare_weight_tracking.py', 'tests/test_weight_tracking.m',
    'tests/bench_weight_tracking.m', 'tests/test_native_prefill.m', 'tools/bench_weight_tracking.py')]
files += sorted((root / 'metal').glob('*.metal'))
files += sorted((root / 'tests/.weight_tracking').glob('*.inc'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {str(p.relative_to(root)): sha(p) for p in files}
for p in files:
    dest = output / 'source_snapshot' / p.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(p.read_bytes())
model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
manifest = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_sha256': hashes, 'binary_sha256': sha(binary), 'binary': str(binary),
    'model': str(model), 'model_size': model.stat().st_size, 'model_mtime_ns': model.stat().st_mtime_ns,
    'method': 'Two replicas; each has depth128, ctx512, 8 warmup and 64 timed greedy AB/BA pairs. Full vocabulary finite and bit exact per forward; final canonical state exact. Private screening, not ordinary native throughput.',
    'cooldown_seconds': args.cooldown, 'runs': []}
write = lambda: (output / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')
write()
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_'))}
env.update(QW3_METAL_KV_F16='1', QW3_METAL_KV_Q8_0='0', QW3_METAL_NGL='64')
try:
    for replica in range(2):
        time.sleep(args.cooldown)
        cmd = [str(binary), str(model)]
        result = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True)
        name = f'replica_{replica + 1}'
        (output / (name + '.stdout.log')).write_text(result.stdout)
        (output / (name + '.stderr.log')).write_text(result.stderr)
        run = {'replica': replica + 1, 'command': cmd, 'exit_code': result.returncode}
        manifest['runs'].append(run)
        write()
        result.check_returncode()
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        if len(rows) != 64 or [int(r['token']) for r in rows] != list(range(8, 72)):
            raise ValueError('missing or duplicate paired samples')
        if any(r['order'] != ('BA' if int(r['token']) & 1 else 'AB') for r in rows):
            raise ValueError('invalid alternating order')
        if 'PASS weight_tracking timed_tokens=64' not in result.stderr:
            raise ValueError('missing numerical validation')
        base = [float(r['baseline_ms']) for r in rows]
        candidate = [float(r['candidate_ms']) for r in rows]
        if any(not math.isfinite(x) or x <= 0 for x in base + candidate):
            raise ValueError('nonpositive durations')
        blocks = [statistics.mean(base[i] - candidate[i] for i in range(b, b + 8)) for b in range(0, 64, 8)]
        run.update(samples=64, gain_pct=100 * (sum(base) / sum(candidate) - 1),
            baseline_tps=64000 / sum(base), candidate_tps=64000 / sum(candidate),
            saved_ms=statistics.mean(blocks), descriptive_95_halfwidth_ms=2.364624 * statistics.stdev(blocks) / (8 ** .5),
            positive_blocks=sum(x > 0 for x in blocks), block_means_ms=blocks)
        write()
        print(json.dumps(run), flush=True)
finally:
    manifest['source_unchanged'] = all(sha(root / name) == digest for name, digest in hashes.items())
    manifest['binary_unchanged'] = sha(binary) == manifest['binary_sha256']
    manifest['model_metadata_unchanged'] = (model.stat().st_size, model.stat().st_mtime_ns) == (manifest['model_size'], manifest['model_mtime_ns'])
    manifest['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write()
if not all(manifest[k] for k in ('source_unchanged', 'binary_unchanged', 'model_metadata_unchanged')):
    raise SystemExit('source, binary or model metadata changed')
