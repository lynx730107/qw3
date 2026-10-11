#!/usr/bin/env python3
"""Sequential production-agent regressions, isolated model tests and live URL oracle."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

archive = Path(__file__).resolve().parent
root = archive.parents[2]
previous = json.loads((archive.parent / "agent_regressions_2026-10-11/manifest.json").read_text())
model = Path(previous["model"])
guard = previous["planned_guard_targets"] + ["test-agent-store"]
model_targets = previous["planned_model_targets"]
commands = [(target, False, ["make", "-j1", target]) for target in guard]
commands += [(target, True, ["make", "-j1", target]) for target in model_targets]
commands += [("live-url-oracle", True, ["python3", "tests/test_agent_llama_reference.py",
                                     "--candidate-only", "--model", str(model)])]

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def store_inventory():
    store = Path("/Users/pasqualecapasso/.qw3/agent_store")
    return {str(p.relative_to(store)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in store.rglob("*") if p.is_file()}

env = {k: v for k, v in os.environ.items() if not k.startswith(("QW3_", "GGML_", "MTL_"))}
env.update({k: v for k, v in previous["environment"].items() if k != "QW3_REPEAT_ARTIFACT_DIR"})
env["QW3_REPEAT_ARTIFACT_DIR"] = str(archive / "repeat_format_artifacts")
# The runtime fingerprint has a FORCE prerequisite, so make -q is not a
# useful freshness gate. Execute its content-aware rule, then verify hashes.
subprocess.run(["make", "-j1", "qw3-agent", "qw3-cli"], cwd=root, env=env, check=True)
assert not any(k.startswith("DYLD_") for k in env)
assert model.stat().st_size == previous["model_size"]
assert model.stat().st_mtime_ns == previous["model_mtime_ns"]
names = list(previous["source_sha256"]) + ["tests/test_agent_store.c", "tests/test_agent_llama_reference.py"]
sources = {name: sha(root / name) for name in names}
snapshots = {}
for name, expected in sources.items():
    matches = [p for folder in ("agent_regressions_2026-10-11", "iq3_hint_2026-10-11",
                                "agent_store_2026-10-11", "agent_web_2026-10-11")
               if (p := archive.parent / folder / "source_snapshot" / name).is_file()
               and sha(p) == expected]
    assert matches, name
    path = matches[0]
    subprocess.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(root))],
                   cwd=root, capture_output=True, check=True)
    snapshots[name] = os.path.relpath(path, archive)
binaries = {name: sha(root / name) for name in previous["binary_sha256"]}
assert binaries == previous["binary_sha256"]
manifest = {
    "started_utc": now(), "reason": "User requested fresh agent regressions, then continued optimization",
    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "source_sha256": sources, "source_snapshot_paths": snapshots, "binary_sha256": binaries,
    "model": str(model), "model_size": model.stat().st_size, "model_mtime_ns": model.stat().st_mtime_ns,
    "environment": {k: v for k, v in env.items() if k.startswith(("QW3_", "MTL_"))},
    "sequential": True, "private_optimization_variables_cleared": True,
    "no_additional_weight_copies": True, "production_binaries_match_previous_suite": True,
    "planned_guard_targets": guard, "planned_model_targets": model_targets,
    "planned_live_url_oracle": True, "store_before": store_inventory(), "runs": [],
}

def save():
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

save()
for target, needs_model, argv in commands:
    command = ["caffeinate", "-disu", *argv]
    print(f"RUN {target} model={needs_model}", flush=True)
    swap_before = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    start = time.monotonic()
    with (archive / (target + ".stdout")).open("w") as out, (archive / (target + ".stderr")).open("w") as err:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    manifest["runs"].append({"target": target, "model_required": needs_model, "command": command,
                             "exit_code": code, "seconds": time.monotonic() - start,
                             "swap_before": swap_before,
                             "swap_after": subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()})
    save()
    print(f"DONE {target} exit={code} seconds={manifest['runs'][-1]['seconds']:.3f}", flush=True)
    if code:
        manifest["failure"] = target
        break
manifest.update(completed_utc=now(), source_unchanged=all(sha(root / n) == h for n, h in sources.items()),
                binary_unchanged=all(sha(root / n) == h for n, h in binaries.items()),
                model_unchanged=model.stat().st_size == manifest["model_size"] and
                                model.stat().st_mtime_ns == manifest["model_mtime_ns"],
                store_after=store_inventory())
manifest["store_unchanged"] = manifest["store_before"] == manifest["store_after"]
manifest["all_passed"] = len(manifest["runs"]) == len(commands) and all(r["exit_code"] == 0 for r in manifest["runs"])
save()
assert manifest["source_unchanged"] and manifest["binary_unchanged"] and manifest["model_unchanged"]
assert manifest["store_unchanged"] and manifest["all_passed"]
print(f"PASS targets={len(commands)} sources=unchanged binaries=unchanged store=unchanged", flush=True)
