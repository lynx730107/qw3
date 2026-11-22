#!/usr/bin/env python3
"""Long-context native validation for Inkling 2 Mini against llama.cpp.

Validates:
  1. Frontier top-k logits parity between qw3-bench (Metal FP16 KV) and
     llama-server (Metal FP16 KV, same GGUF and exact same token IDs) across
     short context, sliding-window wrap (>1024), and optional global-attention
     depths (>8192).
  2. Needle-in-a-haystack retrieval in qw3-cli (Metal FP16 KV, greedy) with
     memory footprint (/usr/bin/time -l) and throughput measurement.
  3. Large-context allocation and short-prompt generation checks at 32k / 64k.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

DEFAULT_MODEL = "../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf"
DEFAULT_LLAMA_SERVER = (
    "/Users/pasqualecapasso/Downloads/llama-inkling-b12588/"
    "llama-b12588-mix-c0c3de8/llama-server"
)

NEEDLE_CODE = "MX-40719-QZ"
NEEDLE_HEADER = (
    f"Leggi questo registro di spedizioni. "
    f"Il codice sigillo della cassa CALCEDONIO è {NEEDLE_CODE}.\n"
)
NEEDLE_FOOTER = (
    "\nDomanda: qual era il codice sigillo della cassa CALCEDONIO? "
    "Rispondi con il codice esatto e basta."
)


def make_haystack_text(n_lines):
    lines = [
        f"Scheda {i:04d}. La cassa ORDINARIA-{i:04d} è in magazzino, "
        f"con controllo programmato al turno {i % 28 + 1}."
        for i in range(1, n_lines + 1)
    ]
    return NEEDLE_HEADER + "\n".join(lines) + NEEDLE_FOOTER


def parse_time_memory(stderr_text):
    info = {}
    rss = re.search(r"(\d+)\s+maximum resident set size", stderr_text)
    peak = re.search(r"(\d+)\s+peak memory footprint", stderr_text)
    if rss:
        info["max_rss_mib"] = round(int(rss.group(1)) / (1024.0 * 1024.0), 2)
    if peak:
        info["peak_footprint_mib"] = round(int(peak.group(1)) / (1024.0 * 1024.0), 2)
    return info


def post_json(url, payload, timeout=1800):
    req = urllib.request.Request(
        url,
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


class LlamaServer:
    def __init__(self, server_bin, model, ctx_size, fa, ubatch, port, log_path):
        self.cmd = [
            str(server_bin),
            "-m", str(model),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-ngl", "99",
            "-ncmoe", "0",
            "-c", str(ctx_size),
            "-fa", fa,
            "-b", "512",
            "-ub", str(ubatch),
            "-ctk", "f16",
            "-ctv", "f16",
            "--fit", "off",
            "--parallel", "1",
            "--ctx-checkpoints", "0",
        ]
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = Path(log_path)
        self.proc = None
        self.log_fp = None

    def __enter__(self):
        self.log_fp = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            self.cmd, stdout=self.log_fp, stderr=subprocess.STDOUT
        )
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early with rc={self.proc.returncode} (see {self.log_path})"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1) as r:
                    if json.load(r).get("status") == "ok":
                        return self
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("llama-server startup timed out")

    def __exit__(self, exc_type, exc, tb):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log_fp:
            self.log_fp.close()

    def tokenize(self, text):
        return post_json(f"{self.base_url}/tokenize", {"content": text, "add_special": False})["tokens"]

    def top_logprobs(self, token_ids, top_k=5):
        res = post_json(
            f"{self.base_url}/completion",
            {
                "prompt": token_ids,
                "n_predict": 1,
                "temperature": 0,
                "n_probs": top_k,
                "cache_prompt": False,
            },
            timeout=1800,
        )
        probs = res["completion_probabilities"][0]
        return {
            "argmax_id": probs["id"],
            "top_logprobs": [
                {"id": item["id"], "logprob": item["logprob"], "token": item.get("token", "")}
                for item in probs["top_logprobs"]
            ],
            "timings": res.get("timings", {}),
        }


def run_frontier_parity(root, model, llama_server, frontiers, out_dir, fa_modes, port):
    """Run qw3-bench frontier logits dump on a deterministic haystack text,
    then query llama-server on the exact same token prefixes."""
    max_frontier = max(frontiers)
    # Build enough haystack lines to comfortably exceed max_frontier tokens (~30 tokens/line).
    n_lines = max(120, max_frontier // 15 + 100)
    prompt_path = out_dir / "frontier_haystack.txt"
    prompt_path.write_text(make_haystack_text(n_lines))

    # Dump frontier logits from qw3-bench one frontier at a time or via a single sweep
    # if frontiers are evenly spaced; here we run one sweep per frontier list using
    # a single incremental qw3-bench run if possible, or direct frontier calls.
    qw3_dir = out_dir / "qw3_frontiers"
    qw3_dir.mkdir(parents=True, exist_ok=True)
    qw3_results = {}

    frontiers = sorted(frontiers)
    ctx_alloc = max_frontier + 16
    cmd = [
        "caffeinate", "-disu",
        "/usr/bin/time", "-l",
        str(root / "qw3-bench"),
        "-m", str(model),
        "--metal",
        "--prompt-file", str(prompt_path),
        "--frontiers", ",".join(str(f) for f in frontiers),
        "--ctx-alloc", str(ctx_alloc),
        "--gen-tokens", "0",
        "--dump-frontier-logits-dir", str(qw3_dir),
    ]
    t0 = time.monotonic()
    res = subprocess.run(cmd, text=True, capture_output=True, check=True)
    (out_dir / "qw3_frontiers_sweep.stdout").write_text(res.stdout)
    (out_dir / "qw3_frontiers_sweep.stderr").write_text(res.stderr)
    sweep_mem = parse_time_memory(res.stderr)
    sweep_wall = round(time.monotonic() - t0, 2)
    for f in frontiers:
        j = json.loads((qw3_dir / f"frontier_{f:06d}.top_logits.json").read_text())
        qw3_results[f] = {"json": j, "memory": sweep_mem, "sweep_wall_s": sweep_wall}

    # Query llama-server for each requested FA mode on the exact same tokenized prefixes.
    comparisons = {}
    ctx_server = max(2048, ((max_frontier + 511) // 512) * 512 + 512)
    ubatch = 128 if ctx_server >= 16384 else 512
    full_text = prompt_path.read_text()

    for fa in fa_modes:
        log_path = out_dir / f"llama_server_fa_{fa}.log"
        fa_entries = []
        with LlamaServer(llama_server, model, ctx_server, fa, ubatch, port, log_path) as srv:
            all_tokens = srv.tokenize(full_text)
            if len(all_tokens) < max_frontier:
                raise RuntimeError(f"tokenized haystack has {len(all_tokens)} < {max_frontier}")
            for f in frontiers:
                q_json = qw3_results[f]["json"]
                token_ids = q_json["token_ids"]
                if token_ids != all_tokens[:f]:
                    raise RuntimeError(f"token IDs differ at frontier {f}")
                llama_res = srv.top_logprobs(token_ids, top_k=5)
                q_top5 = q_json["top_logits"][:5]
                q_ids = [x["id"] for x in q_top5]
                l_ids = [x["id"] for x in llama_res["top_logprobs"]]
                # Compare logprob gap relative to top-1 (shift-invariant to softmax Z).
                q_gaps = [round(q_top5[i]["logit"] - q_top5[0]["logit"], 4) for i in range(5)]
                l_gaps = [
                    round(llama_res["top_logprobs"][i]["logprob"] - llama_res["top_logprobs"][0]["logprob"], 4)
                    for i in range(5)
                ]
                entry = {
                    "frontier": f,
                    "qw3_argmax": q_json["argmax_id"],
                    "llama_argmax": llama_res["argmax_id"],
                    "argmax_match": q_json["argmax_id"] == llama_res["argmax_id"],
                    "qw3_top5_ids": q_ids,
                    "llama_top5_ids": l_ids,
                    "top5_exact_order_match": q_ids == l_ids,
                    "top5_set_overlap": len(set(q_ids) & set(l_ids)),
                    "qw3_top5_logit_gaps": q_gaps,
                    "llama_top5_logprob_gaps": l_gaps,
                    "qw3_memory": qw3_results[f]["memory"],
                    "llama_timings": llama_res["timings"],
                }
                fa_entries.append(entry)
                print(
                    f"[parity fa={fa}] frontier={f}: argmax_match={entry['argmax_match']} "
                    f"top5_exact={entry['top5_exact_order_match']} overlap={entry['top5_set_overlap']}/5 "
                    f"qw3_ids={q_ids} llama_ids={l_ids}",
                    flush=True,
                )
        comparisons[f"fa_{fa}"] = fa_entries

    return comparisons


def run_needle_case(root, model, target_tokens, ctx_size, out_dir, label):
    """Run qw3-cli needle retrieval near target_tokens (using ~30 tok/line estimate)."""
    # Empirically each line is ~31.4 tokens in Inkling BPE.
    n_lines = max(1, int(round((target_tokens - 65) / 31.4)))
    prompt_path = out_dir / f"needle_{label}.txt"
    prompt_path.write_text(make_haystack_text(n_lines))

    cmd = [
        "caffeinate", "-disu",
        "/usr/bin/time", "-l",
        str(root / "qw3-cli"),
        "--metal", "--kv-f16",
        "-m", str(model),
        "--ctx", str(ctx_size),
        "--nothink",
        "--temp", "0",
        "-n", "192",
        "--prompt-file", str(prompt_path),
    ]
    swap_before = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout.strip()
    t0 = time.monotonic()
    res = subprocess.run(cmd, text=True, capture_output=True)
    wall_s = round(time.monotonic() - t0, 2)
    swap_after = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout.strip()

    (out_dir / f"needle_{label}.stdout").write_text(res.stdout)
    (out_dir / f"needle_{label}.stderr").write_text(res.stderr)

    mem = parse_time_memory(res.stderr)
    m = re.search(
        r"qw3: Inkling prefill (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), "
        r"generation (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)",
        res.stderr,
    )
    kv_m = re.search(r"qw3: Inkling KV FP16 on Metal: ([\d.]+) MiB at (\d+) tokens", res.stderr)
    answer = res.stdout.strip()
    entry = {
        "label": label,
        "ctx_size": ctx_size,
        "returncode": res.returncode,
        "wall_s": wall_s,
        "answer": answer,
        "needle_exact_match": answer == NEEDLE_CODE,
        "needle_present": NEEDLE_CODE in answer,
        "prefill_tokens": int(m.group(1)) if m else None,
        "prefill_s": float(m.group(2)) if m else None,
        "prefill_tps": float(m.group(3)) if m else None,
        "gen_tokens": int(m.group(4)) if m else None,
        "gen_s": float(m.group(5)) if m else None,
        "gen_tps": float(m.group(6)) if m else None,
        "kv_mib": float(kv_m.group(1)) if kv_m else None,
        "memory": mem,
        "swap_before": swap_before,
        "swap_after": swap_after,
    }
    print(f"[needle {label}] {json.dumps(entry)}", flush=True)
    return entry


def run_capacity_probe(root, model, ctx_size, out_dir, label):
    """Verify allocation and short-prompt greedy answer at large capacity (32k / 64k)."""
    cmd = [
        "caffeinate", "-disu",
        "/usr/bin/time", "-l",
        str(root / "qw3-cli"),
        "--metal", "--kv-f16",
        "-m", str(model),
        "--ctx", str(ctx_size),
        "--nothink",
        "--temp", "0",
        "-n", "96",
        "-p", "Rispondi solo OK.",
    ]
    swap_before = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout.strip()
    t0 = time.monotonic()
    res = subprocess.run(cmd, text=True, capture_output=True)
    wall_s = round(time.monotonic() - t0, 2)
    swap_after = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout.strip()

    (out_dir / f"capacity_{label}.stdout").write_text(res.stdout)
    (out_dir / f"capacity_{label}.stderr").write_text(res.stderr)

    mem = parse_time_memory(res.stderr)
    m = re.search(
        r"qw3: Inkling prefill (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), "
        r"generation (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)",
        res.stderr,
    )
    kv_m = re.search(r"qw3: Inkling KV FP16 on Metal: ([\d.]+) MiB at (\d+) tokens", res.stderr)
    entry = {
        "label": label,
        "ctx_size": ctx_size,
        "returncode": res.returncode,
        "wall_s": wall_s,
        "answer": res.stdout.strip(),
        "ok_match": res.stdout.strip() == "OK",
        "prefill_tokens": int(m.group(1)) if m else None,
        "prefill_tps": float(m.group(3)) if m else None,
        "gen_tokens": int(m.group(4)) if m else None,
        "gen_tps": float(m.group(6)) if m else None,
        "kv_mib": float(kv_m.group(1)) if kv_m else None,
        "memory": mem,
        "swap_before": swap_before,
        "swap_after": swap_after,
    }
    print(f"[capacity {label}] {json.dumps(entry)}", flush=True)
    return entry


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / DEFAULT_MODEL)
    parser.add_argument("--llama-server", type=Path, default=Path(DEFAULT_LLAMA_SERVER))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frontiers",
        default="128,1024,1032,2048",
        help="Comma list of frontier token lengths for logits parity vs llama-server (empty to skip)",
    )
    parser.add_argument("--fa-modes", default="off,on", help="llama-server -fa modes to compare (e.g. off,on)")
    parser.add_argument(
        "--needles",
        default="1200:2048,2200:3072,8400:9216",
        help="Comma list of target_tokens:ctx_size pairs for qw3-cli needle retrieval (empty to skip)",
    )
    parser.add_argument(
        "--capacities",
        default="32768,65536",
        help="Comma list of context capacities for allocation + short-prompt check (empty to skip)",
    )
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()

    model = args.model.resolve(strict=True)
    llama_server = args.llama_server.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)

    report = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(model),
        "qw3_prefill_batch": os.environ.get("QW3_INKLING_PREFILL_BATCH", "1"),
        "qw3_revision": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "frontier_parity": {},
        "needle_retrieval": [],
        "capacity_probes": [],
    }
    manifest = args.output / "validation.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")

    if args.frontiers.strip():
        frontiers = [int(x.strip()) for x in args.frontiers.split(",") if x.strip()]
        fa_modes = [x.strip() for x in args.fa_modes.split(",") if x.strip()]
        report["frontier_parity"] = run_frontier_parity(
            root, model, llama_server, frontiers, args.output, fa_modes, args.port
        )
        manifest.write_text(json.dumps(report, indent=2) + "\n")

    if args.needles.strip():
        for spec in args.needles.split(","):
            spec = spec.strip()
            if not spec:
                continue
            target_s, ctx_s = spec.split(":")
            target_t, ctx_t = int(target_s), int(ctx_s)
            entry = run_needle_case(
                root, model, target_t, ctx_t, args.output, f"t{target_t}_ctx{ctx_t}"
            )
            report["needle_retrieval"].append(entry)
            manifest.write_text(json.dumps(report, indent=2) + "\n")

    if args.capacities.strip():
        for cap_s in args.capacities.split(","):
            cap_s = cap_s.strip()
            if not cap_s:
                continue
            cap = int(cap_s)
            entry = run_capacity_probe(root, model, cap, args.output, f"ctx{cap}")
            report["capacity_probes"].append(entry)
            manifest.write_text(json.dumps(report, indent=2) + "\n")

    report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    if any(item["returncode"] != 0 or not item["needle_exact_match"]
           for item in report["needle_retrieval"]):
        raise SystemExit("at least one needle check failed")
    if any(item["returncode"] != 0 or not item["ok_match"]
           for item in report["capacity_probes"]):
        raise SystemExit("at least one capacity check failed")


if __name__ == "__main__":
    main()
