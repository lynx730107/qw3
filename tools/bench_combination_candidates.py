#!/usr/bin/env python3
"""Sequential OFF/A/B/A+B controls, retaining every raw pp128 sample."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--order', default='off,residual,off,round,off,both,off')
    p.add_argument('--binary', type=Path, default=root / 'qw3-combination-bench')
    p.add_argument('--job', choices=('pp128', 'tg32_d128'), default='pp128')
    p.add_argument('--repetitions', type=int, default=5)
    p.add_argument('--cooldown', type=float, default=60)
    args = p.parse_args()
    order = args.order.split(',')
    if any(m not in ('off', 'residual', 'round', 'both', 'logits', 'all') for m in order) or order[0] != 'off' or order[-1] != 'off':
        p.error('use OFF-bracketed residual,round,both modes')
    if not 1 <= args.repetitions <= 10 or not 0 <= args.cooldown <= 60:
        p.error('repetitions 1..10, cooldown 0..60')
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
    files = ['qw3.c', 'qw3_metal.m', 'qw3_metal.h', 'qw3.h', 'qw3_bench.c',
             'qw3_runtime_fingerprint.h', 'Makefile', 'tests/test_native_prefill.m',
             'tests/prepare_combination_candidates.py', 'tests/combination_candidates.inc',
             'tests/combination_candidates.metal', 'tests/test_combination_candidates.m',
             'tests/bench_combination_candidates.m', 'tests/.combination_candidates/core.inc',
             'tests/.combination_candidates/bridge.inc', 'tests/.combination_candidates/quality.inc',
             'tools/bench_combination_candidates.py'] + [str(f.relative_to(root)) for f in sorted((root / 'metal').glob('*.metal'))]
    if binary.name == 'qw3-sum8-bench':
        files += ['tests/prepare_sum8_candidates.py', 'tests/sum8_dispatch.inc',
                  'tests/test_sum8_candidates.m', 'tests/bench_sum8_candidates.m']
        files += [str(f.relative_to(root)) for f in sorted((root / 'tests/.sum8_candidates').iterdir()) if f.is_file()]
    sha = lambda f: hashlib.sha256((root / f).read_bytes()).hexdigest()
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    data = dict(started_utc=stamp(), binary=str(binary), binary_sha256=sha(binary),
                source_sha256={f: sha(f) for f in files}, model=str(model),
                model_size=model.stat().st_size, model_mtime_ns=model.stat().st_mtime_ns,
                context=512, kv='f16', gpu_layers=64, threads=4, seed=1,
                device='Apple M5', phase=args.job + ' including complete output logits',
                warmup=True, repetitions=args.repetitions, cooldown_seconds=args.cooldown,
                order=order, runs=[])
    manifest = args.output / 'results.json'
    save = lambda: manifest.write_text(json.dumps(data, indent=2) + '\n')
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_'))}
    env.update(QW3_METAL_KV_F16='1', QW3_METAL_KV_Q8_0='0', QW3_BENCH_PRINT_SAMPLES='1')
    prompt, generated, depth = (128, 0, 0) if args.job == 'pp128' else (0, 32, 128)
    token_count = 128 if args.job == 'pp128' else 32
    for block, mode in enumerate(order):
        print(f'block={block} mode={mode} cooling={args.cooldown}s', flush=True)
        time.sleep(args.cooldown)
        run_env = dict(env)
        if mode != 'off':
            run_env['QW3_PRIVATE_COMBINATION'] = mode
        command = ['caffeinate', '-disu', str(binary), '-m', str(model), '--metal',
                   '--ngl', '64', '--llama-style', '--ctx-alloc', '512', '-p', str(prompt),
                   '-n', str(generated), '-d', str(depth), '-r', str(args.repetitions), '--seed', '1', '-t', '4']
        swap_before = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
        run = subprocess.run(command, env=run_env, capture_output=True, text=True)
        stem = args.output / f'{block}-{mode}-{args.job}'
        stem.with_suffix('.stdout').write_text(run.stdout)
        stem.with_suffix('.stderr').write_text(run.stderr)
        row = dict(block=block, mode=mode, command=command, exit_code=run.returncode,
                   swap_before=swap_before, swap_after=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip(),
                   environment={k: v for k, v in run_env.items() if k.startswith('QW3_')})
        data['runs'].append(row)
        save()
        run.check_returncode()
        metrics = list(csv.DictReader(io.StringIO(run.stdout)))
        assert len(metrics) == 1
        row['metrics'] = metrics[0]
        for k, v in dict(n_prompt=prompt, n_gen=generated, n_depth=depth, reps=args.repetitions).items():
            assert int(metrics[0][k]) == v
        counters = re.search(r'private dispatches pair=0 add=0 gdn=0 residual_norm=(\d+) round_norm=(\d+) round_saved=(\d+)', run.stderr)
        assert counters
        actual = tuple(map(int, counters.groups()))
        forwards = args.repetitions + 1
        expected = (64 * forwards if mode in ('residual', 'both', 'all') else 0,
                    19 * forwards if mode in ('round', 'both', 'all') else 0,
                    19 * forwards if mode in ('round', 'both', 'all') else 0)
        assert actual == expected, (actual, expected)
        row['dispatch_counters'] = dict(zip(('residual_norm', 'round_norm', 'round_saved'), actual))
        head = re.search(r' sum8_head=(\d+)', run.stderr)
        if head:
            row['dispatch_counters']['sum8_head'] = int(head[1])
            assert bool(int(head[1])) == (mode in ('logits', 'all'))
            if args.job == 'pp128' and mode in ('logits', 'all'):
                assert int(head[1]) == forwards
        elif mode in ('logits', 'all'):
            raise ValueError('missing logits candidate counter')
        samples = re.findall(r'qw3-bench: sample (\d+)/(\d+) ([\d.]+) tok/s ([\d.]+) ms', run.stderr)
        assert len(samples) == args.repetitions
        row['samples'] = [dict(repetition=int(i), tok_s=float(t), ms=float(ms)) for i, _, t, ms in samples]
        save()
        print(f'{mode}: {metrics[0]["avg_tps"]} tok/s {metrics[0]["avg_ms"]} ms', flush=True)
    off = [float(r['metrics']['avg_ms']) for r in data['runs'] if r['mode'] == 'off']
    data['baseline_spread_pct'] = 100 * (max(off) / min(off) - 1)
    data['summary'] = {m: dict(mean_tps=token_count * 1000 / statistics.mean(float(r['metrics']['avg_ms']) for r in data['runs'] if r['mode'] == m),
                            pooled_gain_pct=100 * (statistics.mean(off) / statistics.mean(float(r['metrics']['avg_ms']) for r in data['runs'] if r['mode'] == m) - 1)) for m in dict.fromkeys(order)}
    data['bracketed_comparisons'] = []
    for r in data['runs']:
        if r['mode'] == 'off':
            continue
        left = max((o for o in data['runs'] if o['mode'] == 'off' and o['block'] < r['block']), key=lambda o: o['block'])
        right = min((o for o in data['runs'] if o['mode'] == 'off' and o['block'] > r['block']), key=lambda o: o['block'])
        baseline = statistics.mean(float(o['metrics']['avg_ms']) for o in (left, right))
        data['bracketed_comparisons'].append(dict(mode=r['mode'], block=r['block'], off_blocks=[left['block'], right['block']],
                                               gain_pct=100 * (baseline / float(r['metrics']['avg_ms']) - 1)))
    assert all(sha(f) == h for f, h in data['source_sha256'].items())
    assert sha(binary) == data['binary_sha256']
    data.update(completed_utc=stamp(), sources_unchanged=True, binary_unchanged=True)
    save()
    print(json.dumps({k: data[k] for k in ('summary', 'baseline_spread_pct', 'bracketed_comparisons')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
