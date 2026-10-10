#!/usr/bin/env python3
"""Archive native K-stage prefill/decode controls without discarding samples."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
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
    p.add_argument('--stage-k', type=int, choices=(64, 128), required=True)
    p.add_argument('--mode', choices=('iq3', 'iq4', 'both'), required=True)
    p.add_argument('--job', choices=('pp128', 'tg32_d128'), default='pp128')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = root / 'qw3-kstage-prefill-bench'
    model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
    files = ['qw3.c', 'qw3_metal.m', 'qw3_metal.h', 'qw3.h', 'qw3_bench.c',
             'qw3_runtime_fingerprint.h', 'Makefile', 'tests/kstage_candidates.inc',
             'tests/prepare_kstage_candidates.py', 'tests/test_native_prefill.m',
             'tests/test_kstage_prefill.m', 'tests/test_kstage_matrix.m',
             'tests/bench_kstage_prefill.m', 'tools/run_kstage_campaign.py']
    files += [str(f.relative_to(root)) for d in ('metal', 'tests/.kstage_candidates')
              for f in sorted((root / d).iterdir()) if f.is_file()]
    sha = lambda f: hashlib.sha256((root / f).read_bytes()).hexdigest()
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    swap = lambda: subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
    command = ['caffeinate', '-disu', str(binary), str(model), str(args.stage_k),
               args.mode, args.job, '5', '60']
    data = dict(started_utc=stamp(), command=command, source_sha256={f: sha(f) for f in files},
                binary_sha256=sha(binary), model_size=model.stat().st_size,
                model_mtime_ns=model.stat().st_mtime_ns, repetitions=5, cooldown_seconds=60,
                stage_k=args.stage_k, mode=args.mode, job=args.job, context=512,
                kv='f16', gpu_layers=64, threads=4, seed=1, warmup=True,
                method='one engine; fresh session per block; native run_llama_style_case',
                runs=[], swap_before=swap())
    for f in files:
        target = args.output / 'source_snapshot' / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / f, target)
    manifest = args.output / 'results.json'
    save = lambda: manifest.write_text(json.dumps(data, indent=2) + '\n')
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_'))}
    env.update(QW3_METAL_KV_F16='1', QW3_METAL_KV_Q8_0='0')
    stderr = args.output / 'campaign.stderr'
    with stderr.open('w') as err, (args.output / 'campaign.stdout').open('w') as out:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err,
                                text=True, env=env, cwd=root)
        current, swap_before = [], swap()
        for line in proc.stdout:
            out.write(line)
            out.flush()
            if line.startswith('KSTAGE_BEGIN'):
                current, swap_before = [], swap()
                print(line.strip(), flush=True)
            elif line.startswith('KSTAGE_END'):
                _, block, candidate, rc = line.split()
                block, candidate, rc = map(int, (block, candidate, rc))
                if rc or block != len(data['runs']) or candidate != block % 2:
                    raise ValueError('K-stage block failed or order changed')
                metrics = list(csv.DictReader(io.StringIO(''.join(current))))
                if len(metrics) != 1:
                    raise ValueError('expected one native benchmark row')
                row = metrics[0]
                pp = args.job == 'pp128'
                expected = dict(n_prompt=128 if pp else 0, n_gen=0 if pp else 32,
                                n_depth=0 if pp else 128, reps=5)
                if any(int(row[k]) != v for k, v in expected.items()):
                    raise ValueError('benchmark workload changed')
                raw = stderr.read_text().split(f'KSTAGE_BEGIN block={block} ', 1)[1].split('KSTAGE_END', 1)
                samples = re.findall(r'qw3-bench: sample (\d+)/5 ([\d.]+) tok/s ([\d.]+) ms', raw[0])
                counters = re.search(r'iq3=(\d+) iq4=(\d+)', raw[1])
                if len(samples) != 5 or not counters:
                    raise ValueError('missing samples or dispatch counters')
                iq3, iq4 = map(int, counters.groups())
                for count, selected in ((iq3, args.mode in ('iq3', 'both')),
                                        (iq4, args.mode in ('iq4', 'both'))):
                    if bool(count) != bool(candidate and selected):
                        raise ValueError('K-stage candidate dispatch mismatch')
                values = [dict(repetition=int(i), tok_s=float(t), ms=float(ms)) for i, t, ms in samples]
                if any(not math.isfinite(x['ms']) or x['ms'] <= 0 for x in values):
                    raise ValueError('invalid sample')
                data['runs'].append(dict(block=block, candidate=candidate, metrics=row,
                    dispatch_counters=dict(iq3=iq3, iq4=iq4), samples=values,
                    swap_before=swap_before, swap_after=swap()))
                save()
                print(f'block={block} candidate={candidate} {row["avg_tps"]} tok/s', flush=True)
            else:
                current.append(line)
        if proc.wait() or len(data['runs']) != 7:
            raise ValueError('incomplete K-stage campaign')
    means = {mode: statistics.mean(float(r['metrics']['avg_ms']) for r in data['runs']
                                 if r['candidate'] == mode) for mode in (0, 1)}
    off = [float(r['metrics']['avg_ms']) for r in data['runs'] if not r['candidate']]
    count = 128 if args.job == 'pp128' else 32
    data['summary'] = dict(off_tps=count * 1000 / means[0], candidate_tps=count * 1000 / means[1],
                           pooled_gain_pct=100 * (means[0] / means[1] - 1),
                           off_spread_pct=100 * (max(off) / min(off) - 1))
    data['bracketed_comparisons'] = [dict(block=i, gain_pct=100 * (
        statistics.mean(float(data['runs'][j]['metrics']['avg_ms']) for j in (i-1, i+1)) /
        float(data['runs'][i]['metrics']['avg_ms']) - 1)) for i in (1, 3, 5)]
    if any(sha(f) != h for f, h in data['source_sha256'].items()) or sha(binary) != data['binary_sha256']:
        raise ValueError('sources or executable changed during campaign')
    data.update(completed_utc=stamp(), sources_unchanged=True, binary_unchanged=True, samples_discarded=0)
    save()
    print(json.dumps(data['summary']), flush=True)


if __name__ == '__main__':
    main()
