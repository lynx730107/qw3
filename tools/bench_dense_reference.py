#!/usr/bin/env python3
"""Sequential AB/BA engine comparison; ordinary synthetic decode, NOT MTP.

Store exact commands, raw stdout/stderr and every paired result. Both engines
use the same GGUF, full trunk/head Metal offload, F16 KV and ctx256. Synthetic
streams differ between engines; no sampling or greedy acceptance is measured.
"""
import argparse
import csv
import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


def run(args, env=None):
    return subprocess.run(args, env=env, text=True, capture_output=True, check=True)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf")
    parser.add_argument("--llama", type=Path, default=root / "../../llama.cpp/build/bin/llama-bench")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.pairs <= 20 or not 1 <= args.repetitions <= 20:
        parser.error("pairs must be 2..20 and repetitions 1..20")
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.model.resolve(strict=True)
    llama = args.llama.resolve(strict=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QW3_", "GGML_"))}
    env.update(QW3_METAL_KV_F16="1", QW3_METAL_KV_Q8_0="0", QW3_BENCH_PRINT_SAMPLES="1")
    data = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(model), "model_size": model.stat().st_size,
        "model_mtime_ns": model.stat().st_mtime_ns,
        "device": run(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip(),
        "qw3_revision": run(["git", "-C", str(root), "rev-parse", "HEAD"]).stdout.strip(),
        "qw3_diff": run(["git", "-C", str(root), "diff", "--stat"]).stdout,
        "kv": "f16", "context": 256, "full_trunk_and_head_offload": True,
        "mtp": False, "same_token_stream": False,
        "workload": "synthetic fixed input tokens, no sampling",
        "pairs": args.pairs, "repetitions": args.repetitions, "warmup": True,
        "runs": [], "summary": {},
    }
    jobs = [("pp128", 128, 0, 0), ("tg32_d0", 0, 32, 0), ("tg32_d128", 0, 32, 128)]
    # df=1..19 Student-t critical values, two-sided 95% paired intervals.
    critical = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
                2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093]
    manifest = args.output / "results.json"
    for label, pp, tg, depth in jobs:
        timings = {"qw3": [], "llama": []}
        for pair in range(args.pairs):
            for engine in (["qw3", "llama"] if pair % 2 == 0 else ["llama", "qw3"]):
                if engine == "qw3":
                    command = [str(root / "qw3-bench"), "-m", str(model), "--ngl", "64",
                               "--llama-style", "--ctx-alloc", "256", "-p", str(pp),
                               "-n", str(tg), "-d", str(depth), "-r", str(args.repetitions)]
                else:
                    # Request all GPU layers, including the output and optional NextN.
                    command = [str(llama), "-m", str(model), "-ngl", "999", "-ctk", "f16",
                               "-ctv", "f16", "-fa", "on", "-b", "128", "-ub", "128",
                               "-p", str(pp), "-n", str(tg), "-d", str(depth),
                               "-r", str(args.repetitions), "-o", "json", "-v"]
                command = ["caffeinate", "-disu", *command]
                print(f"{label} pair={pair + 1}/{args.pairs} engine={engine}", flush=True)
                result = run(command, env)
                stem = args.output / f"{label}-{pair}-{engine}"
                stem.with_suffix(".stdout").write_text(result.stdout)
                stem.with_suffix(".stderr").write_text(result.stderr)
                if engine == "qw3":
                    rows = list(csv.DictReader(io.StringIO(result.stdout)))
                    if len(rows) != 1 or int(rows[0]["n_depth"]) != depth:
                        raise ValueError("unexpected qw3 benchmark output")
                    seconds = float(rows[0]["avg_ms"]) / 1000
                    raw = rows[0]
                else:
                    rows = json.loads(result.stdout)
                    if len(rows) != 1:
                        raise ValueError("unexpected llama benchmark output")
                    raw = rows[0]
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
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("non-finite or non-positive timing")
                timings[engine].append(seconds)
                data["runs"].append({"job": label, "pair": pair, "engine": engine,
                                     "command": command, "seconds": seconds, "raw": raw})
                manifest.write_text(json.dumps(data, indent=2) + "\n")
        differences = [(a - b) * 1000 for a, b in zip(timings["llama"], timings["qw3"])]
        mean = statistics.mean(differences)
        radius = critical[args.pairs - 1] * statistics.stdev(differences) / math.sqrt(args.pairs)
        data["summary"][label] = {
            "qw3_tps": (pp + tg) / statistics.mean(timings["qw3"]),
            "llama_tps": (pp + tg) / statistics.mean(timings["llama"]),
            "paired_saved_ms": mean, "ci95_radius_ms": radius,
            "lead_confirmed_in_this_series": mean - radius > 0,
        }
        manifest.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({label: data["summary"][label]}), flush=True)
    data["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
