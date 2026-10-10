#!/usr/bin/env python3
"""Sequential AB/BA engine comparison; ordinary synthetic decode, NOT MTP.

Store exact commands, raw stdout/stderr and every paired result. Both engines
use the same GGUF, full trunk/head Metal offload, F16 KV and ctx256. Synthetic
streams differ between engines; no sampling or greedy acceptance is measured.
"""
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
import statistics
import subprocess
import time


def run(args, env=None):
    return subprocess.run(args, env=env, text=True, capture_output=True, check=True)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf")
    parser.add_argument("--llama", type=Path, default=root / "../../llama.cpp/build/bin/llama-bench")
    parser.add_argument("--qw3", type=Path, default=root / "qw3-bench")
    parser.add_argument("--qw3-packed8-scratch", action="store_true",
                        help="enable the combined private candidate in its optional diagnostic binary")
    parser.add_argument("--jobs", default="pp128,tg32_d0,tg32_d128")
    parser.add_argument("--cooldown", type=float, default=40,
                        help="idle seconds before each engine pair (0..60)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.pairs <= 20 or not 1 <= args.repetitions <= 20:
        parser.error("pairs must be 2..20 and repetitions 1..20")
    selected_jobs = args.jobs.split(',')
    if not selected_jobs or len(set(selected_jobs)) != len(selected_jobs) or any(
            job not in ('pp128', 'tg32_d0', 'tg32_d128') for job in selected_jobs):
        parser.error('jobs must be a unique list of pp128,tg32_d0,tg32_d128')
    if not 0 <= args.cooldown <= 60:
        parser.error('cooldown must be 0..60')
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.model.resolve(strict=True)
    llama = args.llama.resolve(strict=True)
    qw3 = args.qw3.resolve(strict=True)
    if args.qw3_packed8_scratch and qw3.name != 'qw3-packed8-scratch-bench':
        parser.error('--qw3-packed8-scratch requires the combined diagnostic executable')
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QW3_", "GGML_"))}
    env.update(QW3_METAL_KV_F16="1", QW3_METAL_KV_Q8_0="0", QW3_BENCH_PRINT_SAMPLES="1")
    data = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(model), "model_size": model.stat().st_size,
        "model_mtime_ns": model.stat().st_mtime_ns,
        "device": run(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip(),
        "qw3_revision": run(["git", "-C", str(root), "rev-parse", "HEAD"]).stdout.strip(),
        "qw3_diff": run(["git", "-C", str(root), "diff", "--stat"]).stdout,
        "binary_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (qw3, llama)},
        "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in ([root / n for n in ('qw3.c', 'qw3_metal.m', 'qw3_bench.c')]
                                    + sorted((root / 'metal').glob('*.metal'))
                                    + [root / n for n in ('tests/scratch_candidates.inc', 'tests/scratch_matmul.inc',
                                        'tests/gdn_prefill_dispatch.inc', 'tests/gdn_prefill_candidates.metal',
                                        'tests/bench_packed8_scratch.m', 'tests/bench_scratch_prefill.m',
                                        'qw3.h', 'qw3_metal.h', 'qw3_runtime_fingerprint.h',
                                        'tools/bench_dense_reference.py') if (root / n).exists()])},
        "llama_checkout_revision": run(["git", "-C", str(root / '../../llama.cpp'), 'rev-parse', 'HEAD']).stdout.strip(),
        "qw3_variant": "packed8+direct_scratch" if args.qw3_packed8_scratch else "ordinary",
        "cooldown_seconds_before_pair": args.cooldown,
        "kv": "f16", "context": 256, "full_trunk_and_head_offload": True,
        "mtp": False, "same_token_stream": False,
        "workload": "synthetic fixed input tokens, no sampling",
        "pairs": args.pairs, "repetitions": args.repetitions, "warmup": True,
        "runs": [], "summary": {},
    }
    jobs = [j for j in [("pp128", 128, 0, 0), ("tg32_d0", 0, 32, 0), ("tg32_d128", 0, 32, 128)]
            if j[0] in selected_jobs]
    # df=1..19 Student-t critical values, two-sided 95% paired intervals.
    critical = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
                2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093]
    manifest = args.output / "results.json"
    manifest.write_text(json.dumps(data, indent=2) + '\n')
    for label, pp, tg, depth in jobs:
        timings = {"qw3": [], "llama": []}
        for pair in range(args.pairs):
            print(f'{label} pair={pair + 1}/{args.pairs} cooling {args.cooldown:g}s', flush=True)
            time.sleep(args.cooldown)
            for engine in (["qw3", "llama"] if pair % 2 == 0 else ["llama", "qw3"]):
                if engine == "qw3":
                    command = [str(qw3), "-m", str(model), "--metal", "--ngl", "64",
                               "--llama-style", "--ctx-alloc", "256", "-p", str(pp),
                               "-n", str(tg), "-d", str(depth), "-r", str(args.repetitions),
                               '--seed', '1', '-t', '4']
                else:
                    # Request all GPU layers, including the output and optional NextN.
                    command = [str(llama), "-m", str(model), "-ngl", "999", "-ctk", "f16",
                               "-ctv", "f16", "-fa", "on", "-b", "128", "-ub", "128",
                               "-p", str(pp), "-n", str(tg), "-d", str(depth),
                               "-r", str(args.repetitions), '-t', '4', "-o", "json", "-v"]
                command = ["caffeinate", "-disu", *command]
                print(f"{label} pair={pair + 1}/{args.pairs} engine={engine}", flush=True)
                run_env = dict(env)
                if engine == 'qw3' and args.qw3_packed8_scratch:
                    run_env['QW3_PRIVATE_PACKED8_SCRATCH'] = '1'
                swap_before = run(['sysctl', 'vm.swapusage']).stdout.strip()
                result = subprocess.run(command, env=run_env, text=True, capture_output=True)
                swap_after = run(['sysctl', 'vm.swapusage']).stdout.strip()
                stem = args.output / f"{label}-{pair}-{engine}"
                stem.with_suffix(".stdout").write_text(result.stdout)
                stem.with_suffix(".stderr").write_text(result.stderr)
                if result.returncode:
                    data['failure'] = {'job': label, 'pair': pair + 1, 'engine': engine,
                                       'command': command, 'returncode': result.returncode,
                                       'stdout': stem.with_suffix('.stdout').name,
                                       'stderr': stem.with_suffix('.stderr').name,
                                       'swap_before': swap_before, 'swap_after': swap_after}
                    data['stopped_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    data['sources_unchanged_at_stop'] = all(
                        hashlib.sha256((root / f).read_bytes()).hexdigest() == h
                        for f, h in data['source_sha256'].items())
                    data['binaries_unchanged_at_stop'] = all(
                        hashlib.sha256(Path(f).read_bytes()).hexdigest() == h
                        for f, h in data['binary_sha256'].items())
                    manifest.write_text(json.dumps(data, indent=2) + '\n')
                    result.check_returncode()
                if engine == "qw3":
                    rows = list(csv.DictReader(io.StringIO(result.stdout)))
                    if len(rows) != 1 or any(int(rows[0][key]) != value for key, value in
                            dict(n_prompt=pp, n_gen=tg, n_depth=depth, reps=args.repetitions).items()):
                        raise ValueError("unexpected qw3 benchmark output")
                    if args.qw3_packed8_scratch:
                        counts = re.search(r'gdn=(\d+) scratch_projection=(\d+) scratch_output=(\d+) copies_skipped=(\d+)', result.stderr)
                        if not counts:
                            raise ValueError('missing combined candidate counters')
                        gdn, projections, output, skipped = map(int, counts.groups())
                        if (pp + depth >= 16 and (not gdn or not projections or not output)) or \
                                skipped != projections + output:
                            raise ValueError('combined candidate was not fully active')
                        if pp + depth < 16 and (gdn or projections or output or skipped):
                            raise ValueError('unexpected candidate dispatch during ordinary depth-zero decode')
                    seconds = float(rows[0]["avg_ms"]) / 1000
                    raw = rows[0]
                    matches = re.findall(r'qw3-bench: sample (\d+)/(\d+) ([\d.]+) tok/s ([\d.]+) ms', result.stderr)
                    if len(matches) != args.repetitions or any(int(i) != n or int(total) != args.repetitions
                            for n, (i, total, _, _) in enumerate(matches, 1)):
                        raise ValueError('missing qw3 repetitions')
                    samples = [{'repetition': int(i), 'tok_s': float(t), 'ms': float(ms)}
                               for i, _, t, ms in matches]
                else:
                    rows = json.loads(result.stdout)
                    if len(rows) != 1:
                        raise ValueError("unexpected llama benchmark output")
                    raw = rows[0]
                    build = {k: raw[k] for k in ('build_commit', 'build_number')}
                    if 'llama_binary_build' in data and data['llama_binary_build'] != build:
                        raise ValueError('llama binary build changed during comparison')
                    data['llama_binary_build'] = build
                    if raw["type_k"] != "f16" or raw["type_v"] != "f16" or raw["n_gpu_layers"] != 999:
                        raise ValueError("llama configuration mismatch")
                    offload = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", result.stderr)
                    if not offload or offload[1] != offload[2] or not re.search(r"offloading output layer to GPU", result.stderr):
                        raise ValueError("llama full trunk/head offload was not confirmed")
                    if (raw["n_prompt"], raw["n_gen"], raw["n_depth"]) != (pp, tg, depth):
                        raise ValueError("llama workload mismatch")
                    if not re.search(r"n_ctx\s*=\s*256\b", result.stderr):
                        raise ValueError("llama ctx256 allocation was not confirmed")
                    seconds = float(raw["avg_ns"]) / 1e9
                    if len(raw['samples_ns']) != args.repetitions or len(raw['samples_ts']) != args.repetitions:
                        raise ValueError('missing llama repetitions')
                    samples = [{'repetition': i, 'tok_s': float(t), 'ms': float(ns) / 1e6}
                               for i, (t, ns) in enumerate(zip(raw['samples_ts'], raw['samples_ns']), 1)]
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("non-finite or non-positive timing")
                timings[engine].append(seconds)
                data["runs"].append({"job": label, "pair": pair, "engine": engine,
                                     "command": command, "seconds": seconds, "raw": raw,
                                     "samples": samples,
                                     'swap_before': swap_before, 'swap_after': swap_after,
                                     'environment': {k: v for k, v in run_env.items() if k.startswith('QW3_')}})
                manifest.write_text(json.dumps(data, indent=2) + "\n")
        differences = [(a - b) * 1000 for a, b in zip(timings["llama"], timings["qw3"])]
        mean = statistics.mean(differences)
        radius = critical[args.pairs - 1] * statistics.stdev(differences) / math.sqrt(args.pairs)
        data["summary"][label] = {
            "qw3_tps": (pp + tg) / statistics.mean(timings["qw3"]),
            "llama_tps": (pp + tg) / statistics.mean(timings["llama"]),
            "paired_saved_ms": mean, "ci95_radius_ms": radius,
            "lead_confirmed_in_this_series": mean - radius > 0,
            "qw3_spread_pct": 100 * (max(timings['qw3']) / min(timings['qw3']) - 1),
            "llama_spread_pct": 100 * (max(timings['llama']) / min(timings['llama']) - 1),
            "both_engines_stable_5pct": all(max(v) / min(v) - 1 <= .05 for v in timings.values()),
        }
        manifest.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({label: data["summary"][label]}), flush=True)
    if any(hashlib.sha256(Path(f).read_bytes()).hexdigest() != h
           for f, h in data['binary_sha256'].items()) or any(
           hashlib.sha256((root / f).read_bytes()).hexdigest() != h
           for f, h in data['source_sha256'].items()):
        raise ValueError('sources or executable changed during comparison')
    data.update(sources_unchanged=True, binaries_unchanged=True, samples_discarded=0)
    data["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
