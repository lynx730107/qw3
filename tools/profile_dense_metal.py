#!/usr/bin/env python3
"""Profile ordinary dense Metal stages; intrusive timings are not throughput benchmarks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
    bench = root / 'qw3-bench'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_'))}
    env.update(QW3_METAL_KV_F16='1', QW3_METAL_KV_Q8_0='0')
    result = {
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model': str(model), 'model_size': model.stat().st_size,
        'model_mtime_ns': model.stat().st_mtime_ns,
        'device': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'context': 512, 'kv': 'f16', 'gpu_layers': 64, 'warmup_excluded': True,
        'source_sha256': {f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                          for f in ('qw3.c', 'qw3_metal.m', 'qw3_bench.c', 'Makefile')},
        'limitations': [
            'Decode profiling synchronizes after every stage in every layer.',
            'Prefill profiling disables combined layer command batching and separates attention/FFN submissions.',
            'CPU submission, waits and GPU work are mixed; stage fractions are diagnostic, not production GPU fractions.',
            'Synthetic input tokens, no greedy or sampling throughput comparison.'],
        'jobs': {}}
    for job, prompt, generated, depth, variable in (
        ('pp128', 128, 0, 0, 'QW3_METAL_DENSE_PREFILL_PROFILE'),
        ('tg32_d128', 0, 32, 128, 'QW3_METAL_DENSE_PROFILE')):
        command = ['caffeinate', '-disu', str(bench), '-m', str(model), '--ngl', '64',
                   '--llama-style', '--ctx-alloc', '512', '-p', str(prompt),
                   '-n', str(generated), '-d', str(depth), '-r', '1', '-t', '4']
        run_env = dict(env, **{variable: '1'})
        run = subprocess.run(command, env=run_env, capture_output=True, text=True, check=True)
        (args.output / f'{job}.stdout').write_text(run.stdout)
        (args.output / f'{job}.stderr').write_text(run.stderr)
        rows = []
        marker = 'qw3 dense prefill layer=' if prompt else 'qw3 dense profile pos='
        for line in run.stderr.splitlines():
            if not line.startswith(marker):
                continue
            rows.append({k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)})
        if prompt:
            if len(rows) != 128 or any(r['tokens'] != 128 or r['ok'] != 1 for r in rows):
                raise ValueError('expected one warmup and one measured 64-layer pp128 profile')
            measured = rows[64:]
            if [r['layer'] for r in measured] != list(range(64)):
                raise ValueError('unexpected prefill layer order')
            totals = {k: sum(r[k] for r in measured)
                      for k in ('attention_ms', 'ffn_batch_ms', 'ms')}
            attention_by_kind = {
                kind: sum(r['attention_ms'] for r in measured if ((int(r['layer']) + 1) % 4 == 0) == full)
                for kind, full in (('gqa', True), ('deltanet', False))}
            summary = {'layer_totals_ms': totals, 'attention_totals_ms_by_kind': attention_by_kind}
        else:
            if len(rows) != 33 or any(r['ok'] != 1 for r in rows):
                raise ValueError('expected one warmup forward and 32 measured decode forwards')
            measured = rows[1:]
            if [r['pos'] for r in measured] != list(range(128, 160)):
                raise ValueError('unexpected decode positions')
            fields = ('embed', 'attn_norm', 'gqa', 'deltanet', 'ffn_norm', 'ffn_gateup', 'ffn_down', 'logits', 'total')
            summary = {'mean_ms_per_forward': {k: statistics.mean(r[k] for r in measured) for k in fields},
                       'min_max_total_ms': [min(r['total'] for r in measured), max(r['total'] for r in measured)]}
        result['jobs'][job] = {'command': command, 'profile_environment': {variable: '1',
                               'QW3_METAL_KV_F16': '1', 'QW3_METAL_KV_Q8_0': '0'},
                               'profile_rows_with_warmup': len(rows), 'measured_rows': measured,
                               'summary': summary}
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        print(job, json.dumps(summary), flush=True)
    result['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
