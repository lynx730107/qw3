"""Verify exact row-geometry logs, ordinary samples, metadata and snapshots."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

screen = Path(__file__).resolve().parent
root = screen.parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_manifest(path):
    manifest = json.loads(path.read_text())
    assert all(manifest[k] for k in ("source_unchanged", "binary_unchanged", "model_unchanged"))
    for name, expected in manifest["source_sha256"].items():
        frozen = (path.parent / manifest["source_snapshot_paths"][name]).resolve()
        assert sha(root / name) == sha(frozen) == expected, name
    for key in ("binary_sha256", "production_binary_sha256"):
        for name, expected in manifest[key].items():
            assert sha(root / name) == expected, name
    st = Path(manifest["model"]).stat()
    assert (st.st_size, st.st_mtime_ns) == (manifest["model_size"], manifest["model_mtime_ns"])
    return manifest

def metadata(log):
    return re.findall(r'iq3_rows_metadata variant=(\d+) thread_width=(\d+) max_threads=(\d+) '
                      r'static_threadgroup_bytes=(\d+) rows_per_simd=(\d+) threads=(\d+)', log)

expected_metadata = [("0", "32", "1024", "1024", "4", "64"),
                     ("1", "32", "1024", "1024", "2", "64"),
                     ("2", "32", "1024", "1024", "8", "64")]
numeric = validate_manifest(screen / "manifest.json")
assert numeric["all_passed"] and len(numeric["runs"]) == 3
assert not (screen / "build.stderr").read_text()
assert numeric["additional_weight_buffer_bytes"] == numeric["additional_session_data_buffers"] == 0
assert numeric["shader_changed"]
assert numeric["variants"] == {"1": "rows_per_simd_2", "2": "rows_per_simd_8"}
initial = validate_manifest(screen / "initial_controller/manifest.json")
assert "all_passed" not in initial and len(initial["runs"]) == 1
assert initial["runs"][0]["exit_code"] == 0
assert initial["runs"][0]["command"][2] == "./qw3-iq3-immutable-test"
assert "PASS iq3_immutable=1 matrix_cases=18 finite_bit_exact=all" in (screen / "initial_controller/matrix_v1.stdout").read_text()
assert not (screen / "screening.stderr").read_text()
assert not (screen / "native_controller.stderr").read_text()
for run in numeric["runs"]:
    assert run["exit_code"] == 0
    log = (screen / (run["name"] + ".stderr")).read_text()
    out = (screen / (run["name"] + ".stdout")).read_text()
    assert "Metal API Validation Enabled" in log
    assert metadata(log) == expected_metadata
    if run["name"].startswith("matrix"):
        assert len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$', out, re.M)) == 18
    else:
        phases = re.findall(r'^PASS iq3_rows_model phase=(\d+) prefix=128 continuation=32 '
                            r'finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) '
                            r'sessions=1 additional_weight_buffer_bytes=0$', out, re.M)
        assert phases == [("0", "0"), ("1", "2336"), ("2", "2336")]

native = validate_manifest(screen / "native/manifest.json")
assert native["order"] == [0, 1, 0, 2, 0] and len(native["runs"]) == 5
assert native["cooldown_seconds_before_every_process"] == 60
assert native["repetitions"] == 3 and native["context"] == 256
assert native["source_sha256"] == numeric["source_sha256"]
means = []
for run in native["runs"]:
    assert run["exit_code"] == 0
    log = (screen / "native" / (run["name"] + ".stderr")).read_text()
    assert "Metal API Validation Enabled" not in log
    assert metadata(log) == expected_metadata
    samples = [{"repetition": int(n), "tps": float(tps), "ms": float(ms)}
               for n, tps, ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms', log)]
    assert samples == run["samples"] and [s["repetition"] for s in samples] == [1, 2, 3]
    assert all(math.isfinite(s[k]) and s[k] > 0 for s in samples for k in ("ms", "tps"))
    calls = re.findall(r'PASS iq3_rows_native variant=(\d+) calls=(\d+) additional_weight_buffer_bytes=0', log)
    assert calls == [(str(run["variant"]), str(run["candidate_calls"]))]
    assert run["candidate_calls"] == (7081 if run["variant"] else 0)
    assert run["environment"]["QW3_PRIVATE_IQ3_ROWS"] == str(run["variant"])
    rows = list(csv.DictReader((screen / "native" / (run["name"] + ".stdout")).read_text().splitlines()))
    assert len(rows) == 1 and rows[0]["test"] == "tg"
    assert all(int(rows[0][k]) == v for k, v in
               {"n_prompt": 0, "n_gen": 32, "n_depth": 128, "reps": 3, "session_bytes": 176761680}.items())
    mean = statistics.mean(s["ms"] for s in samples)
    assert abs(float(rows[0]["avg_ms"]) - mean) <= .006
    means.append(mean)
off = [means[i] for i in (0, 2, 4)]
spread = 100 * (max(off) / min(off) - 1)
comparisons = []
for i in (1, 3):
    reference = (means[i - 1] + means[i + 1]) / 2
    comparisons.append({"block": i, "variant": native["runs"][i]["variant"],
                        "candidate_mean_ms": means[i], "adjacent_off_mean_ms": reference,
                        "point_gain_pct": 100 * (reference / means[i] - 1),
                        "gain_vs_left_off_pct": 100 * (means[i - 1] / means[i] - 1),
                        "gain_vs_right_off_pct": 100 * (means[i + 1] / means[i] - 1)})
swap = [{"stage": stage, **{k: r[k] for k in ("name", "swap_before", "swap_after")}}
        for stage, manifest in (("numeric", numeric), ("native", native)) for r in manifest["runs"]]
store = Path("/Users/pasqualecapasso/.qw3/agent_store")
assert sorted(str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()) == ["history"]
store_observation = json.loads((screen / "store_observation.json").read_text())
assert len(store_observation["files"]) == 1
history = store_observation["files"][0]
assert history["path"] == "history" and history["records"] == 1000
assert (store / "history").stat().st_size == history["bytes"] == 30100
assert (store / "history").stat().st_mtime_ns == history["mtime_ns"]
assert sha(store / "history") == history["sha256"]
analysis = {"adopted": False, "numeric_guarded_matrices": 36, "extra_tail_threadgroup_in_matrix_tests": True,
            "initial_controller_wrong_binary_retained_separately": True,
            "initial_controller_run_is_not_row_geometry_evidence": True,
            "numeric_session_count": 1, "numeric_full_vocab_checks_per_phase": 33,
            "numeric_checkpoint_suite_repeated": False, "native_runs": 5,
            "native_samples_retained": 15, "native_samples_discarded": 0,
            "block_means_ms": means, "off_max_over_min_spread_pct": spread,
            "control_stability_gate_passed": spread <= 5, "comparisons": comparisons,
            "additional_weight_buffer_bytes": 0, "additional_session_data_buffers": 0,
            "native_session_bytes": 176761680, "rows_per_simd_by_variant": [4, 2, 8],
            "threads_per_threadgroup": 64, "threadgroup_grid_bytes": 1024, "swap_observations": swap,
            "all_swap_observations_equal": len({s[k] for s in swap for k in ("swap_before", "swap_after")}) == 1,
            "store_history_only_bytes_at_observation_and_end": history["bytes"],
            "store_history_records": history["records"],
            "store_history_updated_before_this_campaign": True,
            "scope": "screening; each variant has one ON run; adjacent controls are shared, no independent-sample CI"}
(screen / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
print(json.dumps(analysis, indent=2))
