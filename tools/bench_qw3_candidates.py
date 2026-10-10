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
    parser.add_argument('--jobs', default='pp128,tg32_d128')
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--cooldown', type=float, default=40)
    parser.add_argument('--repetitions', type=int, default=1)
    args = parser.parse_args()
    order = args.order.split(',')
    jobs = args.jobs.split(',')
    if not jobs or len(set(jobs)) != len(jobs) or any(j not in ('pp128', 'tg32_d128') for j in jobs):
        parser.error('jobs must be a unique list of pp128,tg32_d128')
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
                (name == 'QW3_PRIVATE_PACKED8_SCRATCH' and value == '1') or
                (name == 'QW3_PRIVATE_MICRO' and value in ('prefill', 'decode', 'pair', 'conv', 'all')) or
                (name in ('QW3_METAL_DENSE_PREFILL_FAST', 'QW3_METAL_DENSE_PREFILL_K64',
                          'QW3_METAL_DENSE_GDN_CACHE', 'QW3_PRIVATE_MODEL_RESIDENCY',
                          'QW3_PRIVATE_WEIGHT_UNTRACKED') and value == '1') or
                (name in ('QW3_PRIVATE_IQ3_METADATA', 'QW3_PRIVATE_METADATA_COMBINED', 'QW3_PRIVATE_RESTORE_COPIES') and value == '1') or
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
                                        + [root / n for n in ('qw3.h', 'qw3_metal.h',
                                             'qw3_runtime_fingerprint.h', 'tools/bench_qw3_candidates.py', 'Makefile',
                                             'tests/prepare_weight_tracking_native.py', 'tests/bench_weight_tracking_native.m',
                                             'tests/.weight_tracking/native_core.inc',
                                             'tests/model_residency_candidates.inc', 'tests/bench_model_residency.m',
                                             'tests/test_model_residency.m', 'tests/prepare_model_residency.py',
                                             'tests/.model_residency/quality.inc',
                                             'tests/gdn_cache_candidates.inc', 'tests/bench_gdn_cache.m',
                                             'tests/test_gdn_cache.m', 'tests/prepare_gdn_cache.py',
                                             'tests/test_native_gdn_cache.m',
                                             'tests/.gdn_cache/kernels.metal', 'tests/.gdn_cache/quality.inc',
                                             'tests/.gdn_cache/native_quality.inc', 'tests/.gdn_cache/bench.inc',
                                             'tests/dense_candidates.inc',
                                             'tests/bench_dense_candidates.m', 'tests/gdn_prefill_candidates.inc',
                                             'tests/gdn_prefill_dispatch.inc',
                                             'tests/gdn_prefill_candidates.metal', 'tests/bench_gdn_prefill.m',
                                             'tests/projection_candidates.inc', 'tests/projection_candidates.metal',
                                             'tests/bench_projection_prefill.m', 'tests/bench_projection_cost.m',
                                             'tests/scratch_candidates.inc', 'tests/scratch_matmul.inc',
                                             'tests/bench_scratch_prefill.m', 'tests/test_scratch_prefill.m',
                                             'tests/test_scratch_matmul.m', 'tests/bench_packed8_scratch.m',
                                             'tests/test_packed8_scratch.m', 'tests/bench_revisit_prefill.m',
                                             'tests/test_revisit_prefill.m', 'tests/native_prefill_validation.inc',
                                             'tests/test_copy_control.m', 'tests/bench_copy_control.m',
                                             'tests/prepare_micro_candidates.py', 'tests/micro_candidates.inc',
                                             'tests/micro_candidates.metal', 'tests/bench_micro_candidates.m',
                                             'tests/test_micro_candidates.m', 'tests/test_micro_q8.m',
                                             'tests/.micro_candidates/core.inc', 'tests/.micro_candidates/quality.inc') if (root / n).exists()])},
            'device': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'context': 512, 'kv': 'f16', 'gpu_layers': 64, 'seed': 1,
            'workload': 'same seed-1 fixed input IDs for each job across all blocks; no sampling',
            'timing': 'qw3-bench pure pp128 and 32 decode forwards at depth128, including output logits',
            'warmup': True, 'repetitions': args.repetitions, 'cooldown_seconds': args.cooldown,
            'candidate_environment': candidate_env, 'jobs': jobs, 'runs': []}
    manifest = args.output / 'results.json'
    def save():
        manifest.write_text(json.dumps(data, indent=2) + '\n')
    save()
    for block, mode in enumerate(order):
        print(f'block={block} mode={mode} cooling {args.cooldown:g}s', flush=True)
        time.sleep(args.cooldown)
        for job, prompt, generated, depth in (('pp128', 128, 0, 0), ('tg32_d128', 0, 32, 128)):
            if job not in jobs:
                continue
            command = ['caffeinate', '-disu', str(binary), '-m', str(model), '--metal',
                       '--ngl', '64', '--llama-style', '--ctx-alloc', '512', '-p', str(prompt),
                       '-n', str(generated), '-d', str(depth), '-r', str(args.repetitions), '--seed', '1', '-t', '4']
            run_env = dict(env, **(candidate_env if mode == 'candidate' else {}))
            if 'QW3_METAL_DENSE_PREFILL_FAST' in candidate_env and mode == 'off':
                run_env['QW3_METAL_DENSE_PREFILL_FAST'] = '0'
            if 'QW3_METAL_DENSE_PREFILL_K64' in candidate_env and mode == 'off':
                run_env['QW3_METAL_DENSE_PREFILL_K64'] = '0'
            if 'QW3_METAL_DENSE_GDN_CACHE' in candidate_env and mode == 'off':
                run_env['QW3_METAL_DENSE_GDN_CACHE'] = '0'
            swap_before = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
            run = subprocess.run(command, env=run_env, capture_output=True, text=True, check=True)
            swap_after = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
            stem = args.output / f'{block}-{mode}-{job}'
            stem.with_suffix('.stdout').write_text(run.stdout)
            stem.with_suffix('.stderr').write_text(run.stderr)
            calls = re.search(r'private dispatches pair=(\d+) add=(\d+) gdn=(\d+)', run.stderr)
            if candidate_env and not any(k in candidate_env for k in
                    ('QW3_METAL_DENSE_PREFILL_FAST', 'QW3_METAL_DENSE_PREFILL_K64', 'QW3_METAL_DENSE_GDN_CACHE')):
                if not calls:
                    raise ValueError('missing private dispatch counters')
                pair, add, gdn = map(int, calls.groups())
                if 'QW3_PRIVATE_MODEL_RESIDENCY' in candidate_env:
                    resident = re.search(r' residency_requests=(\d+) residency_views=(\d+) residency_bytes=(\d+)', run.stderr)
                    if not resident or any(bool(int(v)) != (mode == 'candidate') for v in resident.groups()):
                        raise ValueError('model residency activation mismatch')
                if 'QW3_PRIVATE_WEIGHT_UNTRACKED' in candidate_env:
                    weights = re.search(r' weight_untracked=(\d+) weight_views=(\d+) weight_bytes=(\d+)', run.stderr)
                    if not weights or int(weights[1]) != (mode == 'candidate') or int(weights[2]) != 1 or int(weights[3]) <= 0:
                        raise ValueError('weight view tracking activation mismatch')
                if mode == 'off' and (pair or add or gdn):
                    raise ValueError('candidate dispatched during an OFF control')
                if 'QW3_PRIVATE_MICRO' in candidate_env:
                    micro = re.search(r' prefill_pair=(\d+) decode_pair=(\d+) conv_norm=(\d+)', run.stderr)
                    if micro is None:
                        raise ValueError('missing micro dispatch counters')
                    prefill_pair, decode_pair, conv_norm = map(int, micro.groups())
                    selected = candidate_env['QW3_PRIVATE_MICRO']
                    if mode == 'off' and (prefill_pair or decode_pair or conv_norm):
                        raise ValueError('micro dispatch during OFF')
                    if mode == 'candidate':
                        if selected in ('prefill', 'pair', 'all') and not prefill_pair:
                            raise ValueError('prefill pair did not dispatch')
                        if selected in ('decode', 'pair', 'all') and generated and not decode_pair:
                            raise ValueError('decode pair did not dispatch')
                        if selected == 'prefill' and decode_pair:
                            raise ValueError('decode changed in prefill experiment')
                        if selected == 'decode' and prefill_pair:
                            raise ValueError('prefill changed in decode experiment')
                        if selected in ('conv', 'all') and not conv_norm:
                            raise ValueError('conv norm did not dispatch')
                        if selected != 'all' and ((selected == 'conv' and (prefill_pair or decode_pair)) or
                                                 (selected != 'conv' and conv_norm)):
                            raise ValueError('unexpected micro combination')
                q3 = re.search(r' q3=(\d+)', run.stderr)
                if 'QW3_PRIVATE_Q3_METADATA' in candidate_env:
                    if q3 is None or (mode == 'off' and int(q3[1])) or (mode == 'candidate' and not int(q3[1])):
                        raise ValueError('unexpected private Q3 dispatch count')
                iq = re.search(r' iq=(\d+)', run.stderr)
                if 'QW3_PRIVATE_IQ_METADATA' in candidate_env:
                    if iq is None or (mode == 'off' and int(iq[1])) or (mode == 'candidate' and not int(iq[1])):
                        raise ValueError('unexpected private IQ dispatch count')
                if 'QW3_PRIVATE_DIRECT_SCRATCH' in candidate_env or 'QW3_PRIVATE_PACKED8_SCRATCH' in candidate_env:
                    scratch = re.search(r' scratch_projection=(\d+) scratch_output=(\d+) copies_skipped=(\d+)', run.stderr)
                    if scratch is None:
                        raise ValueError('missing direct-scratch dispatch counters')
                    projections, output, skipped = map(int, scratch.groups())
                    if mode == 'off' and (projections or output or skipped):
                        raise ValueError('direct-scratch candidate executed during OFF')
                    if skipped != projections + output:
                        raise ValueError('direct-scratch copy count disagrees with dispatches')
                    if mode == 'candidate':
                        setting = candidate_env.get('QW3_PRIVATE_DIRECT_SCRATCH', 'all')
                        if setting in ('projections', 'all') and not projections:
                            raise ValueError('direct-scratch projections did not dispatch')
                        if setting in ('output', 'all') and not output:
                            raise ValueError('direct-scratch output did not dispatch')
                if 'QW3_PRIVATE_RESTORE_COPIES' in candidate_env:
                    scratch = re.search(r' scratch_projection=(\d+) scratch_output=(\d+) copies_skipped=(\d+) restore=(\d+)', run.stderr)
                    if scratch is None:
                        raise ValueError('missing copy-control counters')
                    projections, output, skipped, restore = map(int, scratch.groups())
                    if restore != int(mode == 'candidate') or skipped != projections + output:
                        raise ValueError('copy-control mode/count mismatch')
                    if mode == 'off' and (not projections or not output):
                        raise ValueError('combined default was not active')
                    if mode == 'candidate' and (projections or output or skipped):
                        raise ValueError('copy restoration did not disable strided dispatch')
                if any(k in candidate_env for k in ('QW3_PRIVATE_IQ3_METADATA', 'QW3_PRIVATE_METADATA_COMBINED')):
                    iq3 = re.search(r' iq3=(\d+)', run.stderr)
                    iq4 = re.search(r' iq4=(\d+)', run.stderr)
                    if iq3 is None or iq4 is None or int(iq4[1]) or (mode == 'off' and int(iq3[1])) or (mode == 'candidate' and not int(iq3[1])):
                        raise ValueError('unexpected IQ3-only dispatch count')
                    if 'QW3_PRIVATE_METADATA_COMBINED' in candidate_env and (q3 is None or (mode == 'candidate' and not int(q3[1]))):
                        raise ValueError('combined metadata Q3 did not dispatch')
                if mode == 'candidate':
                    if 'QW3_PRIVATE_FFN_SG8' in candidate_env and generated and (not pair or not add):
                        raise ValueError('FFN candidate did not dispatch')
                    if any(k in candidate_env for k in ('QW3_PRIVATE_GDN_GATES', 'QW3_PRIVATE_GDN_LAYOUT', 'QW3_PRIVATE_PACKED8_SCRATCH')) and not gdn:
                        raise ValueError('DeltaNet candidate did not dispatch')
            rows = list(csv.DictReader(io.StringIO(run.stdout)))
            if len(rows) != 1:
                raise ValueError('expected one benchmark row')
            row = rows[0]
            expected = dict(n_prompt=prompt, n_gen=generated, n_depth=depth, reps=args.repetitions)
            if any(int(row[k]) != v for k, v in expected.items()):
                raise ValueError('unexpected counts')
            samples = re.findall(r'qw3-bench: sample (\d+)/(\d+) ([\d.]+) tok/s ([\d.]+) ms', run.stderr)
            if len(samples) != args.repetitions or any(int(i) != n or int(total) != args.repetitions
                    for n, (i, total, _, _) in enumerate(samples, 1)):
                raise ValueError('missing benchmark repetitions')
            data['runs'].append({'block': block, 'mode': mode, 'job': job,
                                 'samples': [{'repetition': int(i), 'tok_s': float(t), 'ms': float(ms)}
                                             for i, _, t, ms in samples],
                                 'command': command, 'swap_before': swap_before, 'swap_after': swap_after, 'environment': {k: v for k, v in run_env.items()
                                     if k.startswith('QW3_')}, 'metrics': row})
            save()
            print(f'{mode} {job} {row["avg_tps"]} tokens/s {row["avg_ms"]} ms', flush=True)
    data['summary'] = {}
    for job, count in (('pp128', 128), ('tg32_d128', 32)):
        if job not in jobs:
            continue
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
    if hashlib.sha256(binary.read_bytes()).hexdigest() != data['binary_sha256'] or any(
            hashlib.sha256((root / f).read_bytes()).hexdigest() != h
            for f, h in data['source_sha256'].items()):
        raise ValueError('sources or executable changed during campaign')
    data.update(sources_unchanged=True, binary_unchanged=True, samples_discarded=0)
    data['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps(data['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
