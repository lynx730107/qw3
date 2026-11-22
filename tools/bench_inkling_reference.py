#!/usr/bin/env python3
"""Repeatable AB/BA Inkling 2 Mini benchmark: qw3-bench vs llama-bench.

Both engines use the exact same GGUF, full Metal offload (all experts on GPU),
FP16 KV cache, and matching (n_prompt, n_gen, n_depth) workloads in balanced
AB/BA order. Each backend generates its own synthetic token stream. Raw
stdout/stderr, process memory footprint (/usr/bin/time -l), swap
usage, and paired summaries are saved into --output.
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

DEFAULT_JOBS = "pp128_d0,tg32_d0,tg32_d128,pp128_d1024,tg32_d1024"

JOB_SPECS = {
    "pp128_d0": (128, 0, 0, 256),
    "pp512_d0": (512, 0, 0, 1024),
    "tg32_d0": (0, 32, 0, 256),
    "tg128_d0": (0, 128, 0, 256),
    "tg32_d128": (0, 32, 128, 256),
    "pp128_d1024": (128, 0, 1024, 2048),
    "tg32_d1024": (0, 32, 1024, 2048),
    "tg128_d1024": (0, 128, 1024, 2048),
    "tg32_d2048": (0, 32, 2048, 4096),
    "tg32_d0_ctx32k": (0, 32, 0, 32768),
    "tg32_d0_ctx64k": (0, 32, 0, 65536),
}

# Two-sided 95% Student-t critical values for df = 1..19.
T_CRITICAL_95 = [
    0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
    2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
]


def run_capture(args, env=None):
    return subprocess.run(args, env=env, text=True, capture_output=True, check=True)


def parse_time_memory(stderr_text):
    info = {}
    rss = re.search(r"(\d+)\s+maximum resident set size", stderr_text)
    peak = re.search(r"(\d+)\s+peak memory footprint", stderr_text)
    if rss:
        info["max_rss_bytes"] = int(rss.group(1))
        info["max_rss_mib"] = round(int(rss.group(1)) / (1024.0 * 1024.0), 2)
    if peak:
        info["peak_footprint_bytes"] = int(peak.group(1))
        info["peak_footprint_mib"] = round(int(peak.group(1)) / (1024.0 * 1024.0), 2)
    ctx_mem = re.search(
        r"qw3-bench: context buffers ([\d.]+) MiB \(ctx=(\d+), backend=(\w+), gqa_kv=([\d.]+) MiB, deltanet=([\d.]+) MiB, scratch=([\d.]+) MiB\)",
        stderr_text,
    )
    if ctx_mem:
        info["qw3_context_total_mib"] = float(ctx_mem.group(1))
        info["qw3_ctx_alloc"] = int(ctx_mem.group(2))
        info["qw3_kv_mib"] = float(ctx_mem.group(4))
        info["qw3_conv_mib"] = float(ctx_mem.group(5))
        info["qw3_scratch_mib"] = float(ctx_mem.group(6))
    return info


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf",
    )
    parser.add_argument(
        "--llama",
        type=Path,
        default=Path("/Users/pasqualecapasso/Downloads/llama-inkling-b12588/llama-b12588-mix-c0c3de8/llama-bench"),
    )
    parser.add_argument("--qw3", type=Path, default=root / "qw3-bench")
    parser.add_argument("--jobs", default=DEFAULT_JOBS, help=f"Comma list from: {','.join(JOB_SPECS)}")
    parser.add_argument("--pairs", type=int, default=2, help="Number of AB/BA pairs (2..20)")
    parser.add_argument("--repetitions", type=int, default=2, help="Timed repetitions per process (1..20)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Idle seconds between runs/pairs (0..60)")
    parser.add_argument("--cooldown-scope", choices=("pair", "run"), default="pair")
    parser.add_argument("--llama-ubatch", type=int, default=512)
    parser.add_argument("--llama-batch", type=int, default=512)
    parser.add_argument("--qw3-prefill-batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 2 <= args.pairs <= 20 or not 1 <= args.repetitions <= 20:
        parser.error("pairs must be 2..20 and repetitions 1..20")
    if not 0.0 <= args.cooldown <= 60.0:
        parser.error("cooldown must be 0..60")
    if not 1 <= args.qw3_prefill_batch <= 128:
        parser.error("qw3-prefill-batch must be 1..128")

    selected = [j.strip() for j in args.jobs.split(",") if j.strip()]
    if not selected or len(set(selected)) != len(selected) or any(j not in JOB_SPECS for j in selected):
        parser.error(f"jobs must be unique names from: {', '.join(JOB_SPECS)}")

    model = args.model.resolve(strict=True)
    llama = args.llama.resolve(strict=True)
    qw3 = args.qw3.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)

    env = {k: v for k, v in os.environ.items() if not k.startswith(("QW3_", "GGML_", "MTL_"))}
    env.update(QW3_METAL_KV_F16="1", QW3_METAL_KV_Q8_0="0", QW3_BENCH_PRINT_SAMPLES="1")
    env["QW3_INKLING_PREFILL_BATCH"] = str(args.qw3_prefill_batch)

    source_files = [
        root / n
        for n in (
            "qw3.c",
            "qw3.h",
            "qw3_inkling_cpu.inc",
            "qw3_metal.m",
            "qw3_metal.h",
            "qw3_bench.c",
            "metal/qw3_core_linear.metal",
            "metal/qw3_core_moe.metal",
            "metal/qw3_core_sequence.metal",
            "tools/bench_inkling_reference.py",
        )
        if (root / n).exists()
    ]

    data = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(model),
        "model_size": model.stat().st_size,
        "device": run_capture(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip(),
        "qw3_revision": run_capture(["git", "-C", str(root), "rev-parse", "HEAD"]).stdout.strip(),
        "qw3_diff_stat": run_capture(["git", "-C", str(root), "diff", "--stat"]).stdout,
        "binary_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (qw3, llama)},
        "source_sha256": {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in source_files
        },
        "kv": "f16",
        "pairs": args.pairs,
        "repetitions": args.repetitions,
        "warmup": not args.no_warmup,
        "cooldown": args.cooldown,
        "cooldown_scope": args.cooldown_scope,
        "order_balanced": args.pairs % 2 == 0,
        "llama_batch": args.llama_batch,
        "llama_ubatch": args.llama_ubatch,
        "qw3_prefill_batch": args.qw3_prefill_batch,
        "threads": args.threads,
        "jobs": selected,
        "runs": [],
        "summary": {},
    }
    manifest = args.output / "results.json"
    manifest.write_text(json.dumps(data, indent=2) + "\n")

    for label in selected:
        pp, tg, depth, ctx_alloc = JOB_SPECS[label]
        timings = {"qw3": [], "llama": []}
        footprints = {"qw3": [], "llama": []}
        for pair in range(args.pairs):
            if args.cooldown > 0 and args.cooldown_scope == "pair":
                time.sleep(args.cooldown)
            order = ["qw3", "llama"] if pair % 2 == 0 else ["llama", "qw3"]
            for engine in order:
                if args.cooldown > 0 and args.cooldown_scope == "run":
                    time.sleep(args.cooldown)
                if engine == "qw3":
                    cmd = [
                        str(qw3),
                        "-m", str(model),
                        "--metal",
                        "--llama-style",
                        "--ctx-alloc", str(ctx_alloc),
                        "-p", str(pp),
                        "-n", str(tg),
                        "-d", str(depth),
                        "-r", str(args.repetitions),
                        "--seed", "1",
                        "-t", str(args.threads),
                    ]
                    if args.no_warmup:
                        cmd.append("--no-warmup")
                else:
                    cmd = [
                        str(llama),
                        "--device", "MTL0",
                        "-m", str(model),
                        "-ngl", "99",
                        "-ncmoe", "0",
                        "-ctk", "f16",
                        "-ctv", "f16",
                        "-fa", "on",
                        "-b", str(args.llama_batch),
                        "-ub", str(args.llama_ubatch),
                        "-p", str(pp),
                        "-n", str(tg),
                        "-d", str(depth),
                        "-r", str(args.repetitions),
                        "-t", str(args.threads),
                        "-o", "json",
                        "-v",
                    ]
                    if args.no_warmup:
                        cmd.append("--no-warmup")

                wrapped = ["caffeinate", "-disu", "/usr/bin/time", "-l", *cmd]
                print(f"[{label}] pair {pair + 1}/{args.pairs} engine={engine}", flush=True)
                swap_before = run_capture(["sysctl", "vm.swapusage"]).stdout.strip()
                t0 = time.monotonic()
                res = subprocess.run(wrapped, env=env, text=True, capture_output=True)
                wall_s = time.monotonic() - t0
                swap_after = run_capture(["sysctl", "vm.swapusage"]).stdout.strip()

                stem = args.output / f"{label}-p{pair}-{engine}"
                stem.with_suffix(".stdout").write_text(res.stdout)
                stem.with_suffix(".stderr").write_text(res.stderr)
                mem = parse_time_memory(res.stderr)

                if res.returncode != 0:
                    data["failure"] = {
                        "job": label,
                        "pair": pair,
                        "engine": engine,
                        "command": wrapped,
                        "returncode": res.returncode,
                        "swap_before": swap_before,
                        "swap_after": swap_after,
                        "memory": mem,
                    }
                    manifest.write_text(json.dumps(data, indent=2) + "\n")
                    res.check_returncode()

                if engine == "qw3":
                    rows = list(csv.DictReader(io.StringIO(res.stdout)))
                    if len(rows) != 1:
                        raise ValueError(f"unexpected qw3 CSV output for {label}: {res.stdout!r}")
                    raw = rows[0]
                    seconds = float(raw["avg_ms"]) / 1000.0
                    matches = re.findall(
                        r"qw3-bench: sample (\d+)/(\d+) ([\d.]+) tok/s ([\d.]+) ms",
                        res.stderr,
                    )
                    samples = [
                        {"repetition": int(i), "tok_s": float(ts), "ms": float(ms)}
                        for i, _, ts, ms in matches
                    ]
                else:
                    rows = json.loads(res.stdout)
                    if len(rows) != 1:
                        raise ValueError(f"unexpected llama-bench JSON output for {label}")
                    raw = rows[0]
                    if raw["type_k"] != "f16" or raw["type_v"] != "f16" or raw["n_cpu_moe"] != 0:
                        raise ValueError("llama-bench configuration mismatch")
                    seconds = float(raw["avg_ns"]) / 1e9
                    samples = [
                        {"repetition": i, "tok_s": float(ts), "ms": float(ns) / 1e6}
                        for i, (ts, ns) in enumerate(zip(raw["samples_ts"], raw["samples_ns"]), 1)
                    ]

                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"invalid seconds={seconds}")

                timings[engine].append(seconds)
                if "peak_footprint_mib" in mem:
                    footprints[engine].append(mem["peak_footprint_mib"])

                data["runs"].append(
                    {
                        "job": label,
                        "pair": pair,
                        "engine": engine,
                        "n_prompt": pp,
                        "n_gen": tg,
                        "n_depth": depth,
                        "ctx_alloc": ctx_alloc,
                        "command": wrapped,
                        "seconds": seconds,
                        "wall_seconds": round(wall_s, 3),
                        "tps": (pp + tg) / seconds,
                        "samples": samples,
                        "memory": mem,
                        "swap_before": swap_before,
                        "swap_after": swap_after,
                        "raw": raw,
                    }
                )
                manifest.write_text(json.dumps(data, indent=2) + "\n")

        tokens = pp + tg
        qw3_tps = tokens / statistics.mean(timings["qw3"])
        llama_tps = tokens / statistics.mean(timings["llama"])
        diffs_ms = [(q - l) * 1000.0 for q, l in zip(timings["qw3"], timings["llama"])]
        mean_diff = statistics.mean(diffs_ms)
        ci95 = (
            T_CRITICAL_95[args.pairs - 1] * statistics.stdev(diffs_ms) / math.sqrt(args.pairs)
            if args.pairs > 1
            else 0.0
        )
        summary_entry = {
            "n_prompt": pp,
            "n_gen": tg,
            "n_depth": depth,
            "qw3_ctx_alloc": ctx_alloc,
            "qw3_tps": round(qw3_tps, 2),
            "llama_tps": round(llama_tps, 2),
            "ratio_qw3_vs_llama": round(qw3_tps / llama_tps, 4),
            "qw3_mean_ms": round(statistics.mean(timings["qw3"]) * 1000.0, 2),
            "llama_mean_ms": round(statistics.mean(timings["llama"]) * 1000.0, 2),
            "paired_delta_ms_qw3_minus_llama": round(mean_diff, 2),
            "ci95_radius_ms": round(ci95, 2),
            "qw3_spread_pct": round(100.0 * (max(timings["qw3"]) / min(timings["qw3"]) - 1.0), 2),
            "llama_spread_pct": round(100.0 * (max(timings["llama"]) / min(timings["llama"]) - 1.0), 2),
            "qw3_peak_footprint_mib_max": max(footprints["qw3"]) if footprints["qw3"] else None,
            "llama_peak_footprint_mib_max": max(footprints["llama"]) if footprints["llama"] else None,
        }
        data["summary"][label] = summary_entry
        manifest.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({label: summary_entry}), flush=True)

    data["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
