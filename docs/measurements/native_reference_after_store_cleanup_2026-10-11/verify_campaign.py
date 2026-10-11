#!/usr/bin/env python3
"""Verify the completed ordinary, balanced comparison and its raw evidence."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

archive = Path(__file__).resolve().parent
root = archive.parents[2]
data = json.loads((archive / "results.json").read_text())
launch = json.loads((archive / "launch.json").read_text())
assert data["completed_utc"] and launch["completed_utc"]
assert launch["exit_code"] == 0
assert data["sources_unchanged"] and data["binaries_unchanged"]
assert data["samples_discarded"] == 0 and "failure" not in data
assert data["qw3_variant"] == "ordinary" and data["qw3_diff"] == ""
assert data["pairs"] == 6 and data["repetitions"] == 5
assert data["order_balanced"] and data["cooldown_scope"] == "run"
assert data["cooldown_seconds_before_engine_run"] == 60
assert data["context"] == 256 and data["kv"] == "f16"
assert data["full_trunk_and_head_offload"] and not data["mtp"]
assert not data["same_token_stream"]
assert launch["dyld_overrides_absent"]
assert launch["llama_direct_shared_libraries_unchanged"]
assert launch["no_private_optimization_enabled"]
assert not launch["additional_weight_copies"]
assert not launch["simultaneous_model_processes"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert data["source_sha256"] == launch["snapshot_sha256"]
assert set(data["source_sha256"]) == set(launch["source_snapshot_paths"])
for name, expected in data["source_sha256"].items():
    snapshot = (archive / launch["source_snapshot_paths"][name]).resolve()
    assert sha(root / name) == sha(snapshot) == expected, name
    subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch",
                    str(snapshot.relative_to(root))], check=True, capture_output=True)
for name, expected in data["binary_sha256"].items():
    assert sha(name) == expected, name
for name, expected in launch["llama_direct_shared_library_sha256"].items():
    assert sha(name) == expected, name
assert len(launch["llama_direct_shared_library_sha256"]) == 8
previous = json.loads((archive.parent / "native_reference_run_cooldown_2026-10-11/results.json").read_text())
assert data["binary_sha256"] == previous["binary_sha256"]
assert data["source_sha256"] == previous["source_sha256"]
assert data["llama_binary_build"] == previous["llama_binary_build"]
model = Path(data["model"])
assert model.stat().st_size == data["model_size"] == previous["model_size"]
assert model.stat().st_mtime_ns == data["model_mtime_ns"] == previous["model_mtime_ns"]

expected_order = [(pair, engine) for pair in range(6)
                  for engine in (["qw3", "llama"] if pair % 2 == 0 else ["llama", "qw3"])]
assert [(r["pair"], r["engine"]) for r in data["runs"]] == expected_order
timings = {"qw3": [], "llama": []}
swap = []
sample_count = 0
for run in data["runs"]:
    assert run["job"] == "tg32_d128"
    reference_command = next(r["command"] for r in previous["runs"]
                             if r["engine"] == run["engine"] and r["job"] == run["job"])
    assert run["command"] == reference_command
    assert run["environment"] == {"QW3_METAL_KV_F16": "1", "QW3_METAL_KV_Q8_0": "0",
                                   "QW3_BENCH_PRINT_SAMPLES": "1"}
    stem = archive / f'tg32_d128-{run["pair"]}-{run["engine"]}'
    stdout = stem.with_suffix(".stdout").read_text()
    stderr = stem.with_suffix(".stderr").read_text()
    samples = run["samples"]
    assert len(samples) == 5
    assert [s["repetition"] for s in samples] == list(range(1, 6))
    assert all(math.isfinite(s[k]) and s[k] > 0 for s in samples for k in ("ms", "tok_s"))
    if run["engine"] == "qw3":
        rows = list(csv.DictReader(io.StringIO(stdout)))
        assert rows == [run["raw"]]
        row = rows[0]
        assert int(row["session_bytes"]) == 176761680
        assert all(int(row[k]) == v for k, v in
                   {"n_prompt": 0, "n_gen": 32, "n_depth": 128, "reps": 5}.items())
        parsed = re.findall(r'qw3-bench: sample (\d+)/(\d+) ([\d.]+) tok/s ([\d.]+) ms', stderr)
        assert len(parsed) == 5
        for s, (i, total, tps, ms) in zip(samples, parsed):
            assert int(i) == s["repetition"] and int(total) == 5
            assert s["tok_s"] == float(tps) and s["ms"] == float(ms)
        assert abs(statistics.mean(s["ms"] for s in samples) - float(row["avg_ms"])) <= .006
        seconds = float(row["avg_ms"]) / 1000
    else:
        rows = json.loads(stdout)
        assert rows == [run["raw"]]
        row = rows[0]
        assert (row["n_prompt"], row["n_gen"], row["n_depth"]) == (0, 32, 128)
        assert row["type_k"] == row["type_v"] == "f16"
        assert row["n_gpu_layers"] == 999 and row["flash_attn"] == 1
        assert row["n_batch"] == row["n_ubatch"] == 128 and row["n_threads"] == 4
        offload = re.search(r'offloaded\s+(\d+)/(\d+)\s+layers', stderr)
        assert offload and offload[1] == offload[2]
        assert "offloading output layer to GPU" in stderr
        assert re.search(r'n_ctx\s*=\s*256\b', stderr)
        assert len(row["samples_ns"]) == len(row["samples_ts"]) == 5
        for s, ns, tps in zip(samples, row["samples_ns"], row["samples_ts"]):
            assert s["ms"] == ns / 1e6 and s["tok_s"] == tps
        assert abs(statistics.mean(row["samples_ns"]) - row["avg_ns"]) <= 1
        seconds = row["avg_ns"] / 1e9
    assert run["seconds"] == seconds
    timings[run["engine"]].append(seconds)
    swap.extend([run["swap_before"], run["swap_after"]])
    sample_count += len(samples)

differences = [(a - b) * 1000 for a, b in zip(timings["llama"], timings["qw3"])]
mean = statistics.mean(differences)
radius = 2.571 * statistics.stdev(differences) / math.sqrt(6)
qw3_tps = 32 / statistics.mean(timings["qw3"])
llama_tps = 32 / statistics.mean(timings["llama"])
spreads = {k: 100 * (max(v) / min(v) - 1) for k, v in timings.items()}
summary = data["summary"]["tg32_d128"]
for key, expected in {"qw3_tps": qw3_tps, "llama_tps": llama_tps,
                      "paired_saved_ms": mean, "ci95_radius_ms": radius,
                      "qw3_spread_pct": spreads["qw3"], "llama_spread_pct": spreads["llama"],
                      "saved_ms_qw3_first_mean": statistics.mean(differences[::2]),
                      "saved_ms_qw3_second_mean": statistics.mean(differences[1::2])}.items():
    assert math.isclose(summary[key], expected, rel_tol=1e-12, abs_tol=1e-9), key
stable = all(s <= 5 for s in spreads.values())
assert summary["both_engines_stable_5pct"] == stable
assert summary["lead_confirmed_in_this_series"] == (mean - radius > 0)
swap_values = sorted(set(swap))
swap_used_mib = [float(re.search(r'used = ([\d.]+)M', value)[1]) for value in swap]
swap_constant_within_runs = all(r["swap_before"] == r["swap_after"] for r in data["runs"])
assert swap_constant_within_runs
store = Path("/Users/pasqualecapasso/.qw3/agent_store")
assert sorted(str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()) == ["history"]
assert (store / "history").stat().st_size == launch["history_only_store_bytes"] == 30109
assert sha(store / "history") == "3f2fe63f82730e87e86be76795bd335884effe2552ccbc309f4c4d28a8084948"

analysis = {
    "validated_runs": len(data["runs"]), "validated_samples": sample_count,
    "all_recorded_sources_snapshots_and_binaries_match": True,
    "same_sources_and_binaries_as_previous_balanced_comparison": True,
    "eight_direct_llama_libraries_unchanged": True,
    "model_size_mtime_unchanged": True,
    "paired_saved_ms": differences, "rate_delta_pct": 100 * (qw3_tps / llama_tps - 1),
    "qw3_speedup_needed_pct": 100 * (llama_tps / qw3_tps - 1),
    "paired_saved_ci95_ms": [mean - radius, mean + radius],
    "llama_lead_confirmed_in_this_series": stable and mean + radius < 0,
    "qw3_lead_confirmed_in_this_series": stable and mean - radius > 0,
    "both_engines_stable_5pct": stable,
    "swap_before_after_all_runs_equal": len(swap_values) == 1,
    "swap_before_after_each_run_equal": swap_constant_within_runs,
    "swap_usage_values": swap_values,
    "swap_used_delta_mib": swap_used_mib[-1] - swap_used_mib[0],
    "swap_used_never_increased_at_recorded_boundaries": all(
        a >= b for a, b in zip(swap_used_mib, swap_used_mib[1:])),
    "qw3_first_mean_saved_ms": statistics.mean(differences[::2]),
    "qw3_second_mean_saved_ms": statistics.mean(differences[1::2]),
    "store_history_only_bytes_unchanged": 30109,
    "qw3_session_bytes_all_runs": 176761680,
    "interval_experimental_units": "six paired engine-run means; five repetitions each",
    "samples_discarded": 0,
}
(archive / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
print(json.dumps({"summary": summary, "analysis": analysis}, indent=2))
