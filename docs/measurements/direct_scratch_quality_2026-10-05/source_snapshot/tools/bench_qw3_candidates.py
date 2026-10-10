#!/usr/bin/env python3
"""Sequential ordinary pp128/tg32 controls with identical fixed input IDs."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=root / 'qw3-bench')
    parser.add_argument('--order', default='off,off,off,off')
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--cooldown', type=float, default=40)
    parser.add_argument('--repetitions', type=int, default=1)
    args = parser.parse_args()
    order = args.order.split(',')
    if any(c not in ('off', 'candidate') for c in order) or order.count('off') < 2:
        parser.error('include at least two OFF blocks; modes: off,candidate')
    if 'candidate' in order and (order[0] != 'off' or order[-1] != 'off'):
        parser.error('bracket all candidate blocks with OFF controls')
    if not 0 <= args.cooldown <= 60:
        parser.error('cooldown must be 0..60')
    if not 1 <= args.repetitions <= 10:
        parser.error('repetitions must be 1..10')
    candidate_env = {}
    for entry in args.candidate_env:
        if '=' not in entry:
            parser.error('candidate environment entries must be NAME=VALUE')
        name, value = entry.split('=', 1)
        if not ((name in ('QW3_PRIVATE_FFN_SG8', 'QW3_PRIVATE_GDN_GATES') and value == '1') or
                (name == 'QW3_PRIVATE_GDN_LAYOUT' and value in ('packed4', 'packed8', 'pipeline', 'packed8_pipeline')) or
                (name == 'QW3_PRIVATE_DIRECT_SCRATCH' and value in ('projections', 'output', 'all')) or
                (name in ('QW3_PRIVATE_Q3_METADATA', 'QW3_PRIVATE_IQ_METADATA', 'QW3_PRIVATE_STAGE_PROFILE') and value == '1')):
            parser.error('use one documented private candidate setting')
        candidate_env[name] = value
    if len(candidate_env) > 1:
        parser.error('evaluate one candidate at a time')
    if 'candidate' in order and not candidate_env:
        parser.error('candidate blocks require an explicit private setting')
    args.output.mkdir(parents=True, exist_ok=False)
    model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_'))}
    env.update(QW3_METAL_KV_F16='1', QW3_METAL_KV_Q8_0='0', QW3_BENCH_PRINT_SAMPLES='1')
    data = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model': str(model), 'model_size': model.stat().st_size,
            'model_mtime_ns': model.stat().st_mtime_ns,
            'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'source_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in ([root / n for n in ('qw3.c', 'qw3_metal.m', 'qw3_bench.c')]
                                        + sorted((root / 'metal').glob('*.metal'))
                                        + [root / n for n in ('tests/dense_candidates.inc',
                                             'tests/bench_dense_candidates.m', 'tests/gdn_prefill_candidates.inc',
                                             'tests/gdn_prefill_candidates.metal', 'tests/bench_gdn_prefill.m',
                                             'tests/projection_candidates.inc', 'tests/projection_candidates.metal',
                                             'tests/bench_projection_prefill.m', 'tests/bench_projection_cost.m',
                                             'tests/scratch_candidates.inc', 'tests/scratch_matmul.inc',
                                             'tests/bench_scratch_prefill.m', 'tests/test_scratch_prefill.m') if (root / n).exists()])},
            'device': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'context': 512, 'kv': 'f16', 'gpu_layers': 64, 'seed': 1,
            'workload': 'same seed-1 fixed input IDs for each job across all blocks; no sampling',
            'timing': 'qw3-bench pure pp128 and 32 decode forwards at depth128, including output logits',
            'warmup': True, 'repetitions': args.repetitions, 'cooldown_seconds': args.cooldown,
            'candidate_environment': candidate_env, 'runs': []}
    manifest = args.output / 'results.json'
    def save():
        manifest.write_text(json.dumps(data, indent=2) + '\n')
    save()
    for block, mode in enumerate(order):
        print(f'block={block} mode={mode} cooling {args.cooldown:g}s', flush=True)
        time.sleep(args.cooldown)
        for job, prompt, generated, depth in (('pp128', 128, 0, 0), ('tg32_d128', 0, 32, 128)):
            command = ['caffeinate', '-disu', str(binary), '-m', str(model), '--metal',
                       '--ngl', '64', '--llama-style', '--ctx-alloc', '512', '-p', str(prompt),
                       '-n', str(generated), '-d', str(depth), '-r', str(args.repetitions), '--seed', '1', '-t', '4']
            run_env = dict(env, **(candidate_env if mode == 'candidate' else {}))
            swap_before = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
            run = subprocess.run(command, env=run_env, capture_output=True, text=True, check=True)
            swap_after = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
            stem = args.output / f'{block}-{mode}-{job}'
            stem.with_suffix('.stdout').write_text(run.stdout)
            stem.with_suffix('.stderr').write_text(run.stderr)
            calls = re.search(r'private dispatches pair=(\d+) add=(\d+) gdn=(\d+)', run.stderr)
            if candidate_env:
                if not calls:
                    raise ValueError('missing private dispatch counters')
                pair, add, gdn = map(int, calls.groups())
                if mode == 'off' and (pair or add or gdn):
                    raise ValueError('candidate dispatched during an OFF control')
                q3 = re.search(r' q3=(\d+)', run.stderr)
                if 'QW3_PRIVATE_Q3_METADATA' in candidate_env:
                    if q3 is None or (mode == 'off' and int(q3[1])) or (mode == 'candidate' and not int(q3[1])):
                        raise ValueError('unexpected private Q3 dispatch count')
                iq = re.search(r' iq=(\d+)', run.stderr)
                if 'QW3_PRIVATE_IQ_METADATA' in candidate_env:
                    if iq is None or (mode == 'off' and int(iq[1])) or (mode == 'candidate' and not int(iq[1])):
                        raise ValueError('unexpected private IQ dispatch count')
                if 'QW3_PRIVATE_DIRECT_SCRATCH' in candidate_env:
                    scratch = re.search(r' scratch_projection=(\d+) scratch_output=(\d+) copies_skipped=(\d+)', run.stderr)
                    if scratch is None:
                        raise ValueError('missing direct-scratch dispatch counters')
                    projections, output, skipped = map(int, scratch.groups())
                    if mode == 'off' and (projections or output or skipped):
                        raise ValueError('direct-scratch candidate executed during OFF')
                    if skipped != projections + output:
                        raise ValueError('direct-scratch copy count disagrees with dispatches')
                    if mode == 'candidate':
                        setting = candidate_env['QW3_PRIVATE_DIRECT_SCRATCH']
                        if setting in ('projections', 'all') and not projections:
                            raise ValueError('direct-scratch projections did not dispatch')
                        if setting in ('output', 'all') and not output:
                            raise ValueError('direct-scratch output did not dispatch')
                if mode == 'candidate':
                    if 'QW3_PRIVATE_FFN_SG8' in candidate_env and generated and (not pair or not add):
                        raise ValueError('FFN candidate did not dispatch')
                    if any(k in candidate_env for k in ('QW3_PRIVATE_GDN_GATES', 'QW3_PRIVATE_GDN_LAYOUT')) and not gdn:
                        raise ValueError('DeltaNet candidate did not dispatch')
            rows = list(csv.DictReader(io.StringIO(run.stdout)))
            if len(rows) != 1:
                raise ValueError('expected one benchmark row')
            row = rows[0]
            expected = dict(n_prompt=prompt, n_gen=generated, n_depth=depth, reps=args.repetitions)
            if any(int(row[k]) != v for k, v in expected.items()):
                raise ValueError('unexpected counts')
            data['runs'].append({'block': block, 'mode': mode, 'job': job,
                                 'command': command, 'swap_before': swap_before, 'swap_after': swap_after, 'environment': {k: v for k, v in run_env.items()
                                     if k.startswith('QW3_')}, 'metrics': row})
            save()
            print(f'{mode} {job} {row["avg_tps"]} tokens/s {row["avg_ms"]} ms', flush=True)
    data['summary'] = {}
    for job, count in (('pp128', 128), ('tg32_d128', 32)):
        modes = {}
        for mode in dict.fromkeys(order):
            ms = [float(r['metrics']['avg_ms']) for r in data['runs'] if r['job'] == job and r['mode'] == mode]
            modes[mode] = {'mean_tps': count * 1000 / statistics.mean(ms), 'ms_samples': ms}
        off = modes['off']['ms_samples']
        drift = max(off) / min(off) - 1
        data['summary'][job] = {'modes': modes, 'baseline_spread_pct': 100 * drift,
                                'stable_baseline_5pct': drift <= .05}
        if 'candidate' in modes:
            data['summary'][job]['pooled_gain_pct'] = 100 * (statistics.mean(off) /
                         statistics.mean(modes['candidate']['ms_samples']) - 1)
    data['all_baselines_stable_5pct'] = all(s['stable_baseline_5pct'] for s in data['summary'].values())
    data['bracketed_comparisons'] = []
    for row in data['runs']:
        if row['mode'] != 'candidate':
            continue
        controls = [r for r in data['runs'] if r['mode'] == 'off' and r['job'] == row['job']]
        before = [r for r in controls if r['block'] < row['block']]
        after = [r for r in controls if r['block'] > row['block']]
        if not before or not after:
            raise ValueError('candidate blocks must have preceding and following OFF controls')
        left, right = max(before, key=lambda r: r['block']), min(after, key=lambda r: r['block'])
        baseline_ms = statistics.mean(float(r['metrics']['avg_ms']) for r in (left, right))
        data['bracketed_comparisons'].append({'block': row['block'], 'job': row['job'],
            'off_blocks': [left['block'], right['block']], 'baseline_mean_ms': baseline_ms,
            'candidate_ms': float(row['metrics']['avg_ms']),
            'gain_pct': 100 * (baseline_ms / float(row['metrics']['avg_ms']) - 1)})
    data['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps(data['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
