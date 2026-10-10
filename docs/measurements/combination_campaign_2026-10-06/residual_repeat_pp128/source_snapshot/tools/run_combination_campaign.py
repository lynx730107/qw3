#!/usr/bin/env python3
"""Archive a sequential one-engine campaign using native benchmark cases."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--order', required=True)
    p.add_argument('--job', choices=('pp128', 'tg32_d128'), default='pp128')
    args = p.parse_args()
    order = args.order.split(',')
    assert order[0] == order[-1] == 'off'
    assert all(m in ('off', 'residual', 'round', 'both', 'logits', 'all') for m in order)
    args.output.mkdir(parents=True, exist_ok=False)
    binary = root / 'qw3-combination-campaign'
    model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
    files = ['qw3.c', 'qw3_metal.m', 'qw3_metal.h', 'qw3.h', 'qw3_bench.c', 'qw3_runtime_fingerprint.h',
             'Makefile', 'tests/test_native_prefill.m', 'tests/prepare_combination_candidates.py',
             'tests/combination_candidates.inc', 'tests/combination_candidates.metal',
             'tests/prepare_sum8_candidates.py', 'tests/sum8_dispatch.inc',
             'tests/bench_combination_campaign.m', 'tools/run_combination_campaign.py']
    files += [str(f.relative_to(root)) for d in ('metal', 'tests/.combination_candidates', 'tests/.sum8_candidates')
              for f in sorted((root / d).iterdir()) if f.is_file()]
    sha = lambda f: hashlib.sha256((root / f).read_bytes()).hexdigest()
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    command = ['caffeinate', '-disu', str(binary), str(model), args.order, args.job, '5', '60']
    data = dict(started_utc=stamp(), command=command, source_sha256={f: sha(f) for f in files},
                binary_sha256=sha(binary), model_size=model.stat().st_size, model_mtime_ns=model.stat().st_mtime_ns,
                method='one loaded engine; native run_llama_style_case; one fresh session per block',
                phase=args.job, device='Apple M5', context=512, kv='f16', gpu_layers=64, threads=4, seed=1,
                repetitions=5, cooldown_seconds=60, warmup=True, order=order, runs=[],
                swap_before=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip())
    for f in files:
        target = args.output / 'source_snapshot' / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / f, target)
    manifest = args.output / 'results.json'
    save = lambda: manifest.write_text(json.dumps(data, indent=2) + '\n')
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_'))}
    env.update(QW3_METAL_KV_F16='1', QW3_METAL_KV_Q8_0='0', QW3_BENCH_PRINT_SAMPLES='1')
    stderr = args.output / 'campaign.stderr'
    stdout = args.output / 'campaign.stdout'
    with stderr.open('w') as err, stdout.open('w') as out:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, text=True, env=env, cwd=root)
        current = []
        for line in proc.stdout:
            out.write(line)
            out.flush()
            if line.startswith('COMBO_BEGIN'):
                current = []
                print(line.strip(), flush=True)
            elif line.startswith('COMBO_END'):
                _, block, mode, rc = line.split()
                assert int(rc) == 0
                metrics = list(csv.DictReader(io.StringIO(''.join(current))))
                assert len(metrics) == 1
                block = int(block)
                assert mode == order[block] and block == len(data['runs'])
                raw = stderr.read_text().split(f'COMBO_BEGIN {block} {mode}\n', 1)[1].split('COMBO_END', 1)[0]
                samples = re.findall(r'qw3-bench: sample (\d+)/5 ([\d.]+) tok/s ([\d.]+) ms', raw)
                assert len(samples) == 5
                counter = re.search(r'residual_norm=(\d+) round_norm=(\d+) round_saved=(\d+) sum8_head=(\d+)', raw)
                assert counter
                counters = tuple(map(int, counter.groups()))
                expected = (384 if mode in ('residual', 'both', 'all') else 0,
                            114 if mode in ('round', 'both', 'all') else 0,
                            114 if mode in ('round', 'both', 'all') else 0,
                            (6 if args.job == 'pp128' else 167) if mode in ('logits', 'all') else 0)
                assert counters == expected, (mode, counters, expected)
                row = dict(block=block, mode=mode, metrics=metrics[0],
                           dispatch_counters=dict(zip(('residual_norm', 'round_norm', 'round_saved', 'sum8_head'), counters)),
                           samples=[dict(repetition=int(i), tok_s=float(t), ms=float(ms)) for i, t, ms in samples],
                           swap_after=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip())
                data['runs'].append(row)
                stem = args.output / f'{block}-{mode}-{args.job}'
                stem.with_suffix('.stdout').write_text(''.join(current))
                stem.with_suffix('.stderr').write_text(raw)
                save()
                print(f'{mode}: {metrics[0]["avg_tps"]} tok/s, {metrics[0]["avg_ms"]} ms', flush=True)
            else:
                current.append(line)
        assert proc.wait() == 0
    assert len(data['runs']) == len(order)
    off = [float(r['metrics']['avg_ms']) for r in data['runs'] if r['mode'] == 'off']
    count = 128 if args.job == 'pp128' else 32
    data['baseline_spread_pct'] = 100 * (max(off) / min(off) - 1)
    data['summary'] = {mode: dict(mean_tps=count * 1000 / statistics.mean(float(r['metrics']['avg_ms']) for r in data['runs'] if r['mode'] == mode),
                                pooled_gain_pct=100 * (statistics.mean(off) / statistics.mean(float(r['metrics']['avg_ms']) for r in data['runs'] if r['mode'] == mode) - 1)) for mode in dict.fromkeys(order)}
    data['bracketed_comparisons'] = []
    for r in data['runs']:
        if r['mode'] == 'off':
            continue
        left = max((o for o in data['runs'] if o['mode'] == 'off' and o['block'] < r['block']), key=lambda o: o['block'])
        right = min((o for o in data['runs'] if o['mode'] == 'off' and o['block'] > r['block']), key=lambda o: o['block'])
        mean = statistics.mean(float(o['metrics']['avg_ms']) for o in (left, right))
        data['bracketed_comparisons'].append(dict(mode=r['mode'], block=r['block'], off_blocks=[left['block'], right['block']],
                                               gain_pct=100 * (mean / float(r['metrics']['avg_ms']) - 1)))
    assert all(sha(f) == h for f, h in data['source_sha256'].items())
    assert sha(binary) == data['binary_sha256']
    data.update(completed_utc=stamp(), sources_unchanged=True, binary_unchanged=True, samples_discarded=0)
    save()
    print(json.dumps({k: data[k] for k in ('summary', 'baseline_spread_pct', 'bracketed_comparisons')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
