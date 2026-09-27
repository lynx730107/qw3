#!/usr/bin/env python3
"""Live-internet agent/tool quality comparison against llama-server."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REFERENCE_URL = "https://api.github.com/repos/ggerganov/llama.cpp"
REFERENCE_FIELDS = ("full_name", "default_branch", "license", "archived")


def fail(message, details=None):
    print(f"test-agent-llama-reference: FAIL: {message}", file=sys.stderr)
    if details:
        print(details, file=sys.stderr)
    raise SystemExit(1)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "qw3-agent-reference-test/1.0",
        },
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False, f"llama-server exited with status {process.returncode}"
        try:
            request_json(base_url + "/health", timeout=2)
            return True, ""
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            last_error = str(exc)
            time.sleep(0.25)
    return False, last_error


def parse_tool_arguments(call):
    function = call.get("function") or {}
    raw = function.get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    return json.loads(raw or "{}")


def fetch_text(url, timeout):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "qw3-agent-reference-test/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def run_llama_reference(server, model, url, prompt, workdir, timeout):
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = workdir / "llama-server.log"
    log_file = log_path.open("wb")
    command = [
        server,
        "-m", str(model),
        "-ngl", "99",
        "--ctx-size", "8192",
        "--parallel", "1",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--reasoning", "off",
        "--no-webui",
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        ready, error = wait_for_server(base_url, process, timeout)
        if not ready:
            log_file.flush()
            details = error
            if log_path.exists():
                details += "\n" + log_path.read_text(
                    encoding="utf-8", errors="replace")[-6000:]
            fail("llama-server did not become ready", details)
        ready_seconds = time.monotonic() - started
        tools = [{
            "type": "function",
            "function": {
                "name": "visit_url",
                "description": "Fetch one HTTPS URL and return its UTF-8 response body.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        }]
        messages = [
            {"role": "system", "content": "Follow tool results exactly. Never invent records."},
            {"role": "user", "content": prompt},
        ]
        inference_wall = 0.0
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        tool_rounds = 0
        tool_calls = 0
        final_text = ""
        for _ in range(4):
            payload = {
                "model": "local",
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0,
                "seed": 1,
                "max_tokens": 256,
            }
            request_started = time.monotonic()
            response = request_json(
                base_url + "/v1/chat/completions", payload, timeout=timeout)
            inference_wall += time.monotonic() - request_started
            current_usage = response.get("usage") or {}
            for key in usage:
                usage[key] += int(current_usage.get(key) or 0)
            choices = response.get("choices") or []
            if not choices:
                fail("llama-server returned no choices", json.dumps(response)[:2000])
            message = choices[0].get("message") or {}
            calls = message.get("tool_calls") or []
            if not calls:
                final_text = message.get("content") or ""
                break
            tool_rounds += 1
            tool_calls += len(calls)
            messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls,
            })
            for call in calls:
                function = call.get("function") or {}
                if function.get("name") != "visit_url":
                    fail("llama-server selected an unexpected tool", json.dumps(call))
                arguments = parse_tool_arguments(call)
                requested = arguments.get("url", "")
                if requested != url:
                    result = "error: URL outside the permitted reference endpoint"
                else:
                    result = fetch_text(url, timeout)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "read-call"),
                    "content": result,
                })
        if not final_text:
            fail("llama-server did not produce a final answer")
        return {
            "output": final_text,
            "tool_rounds": tool_rounds,
            "tool_calls": tool_calls,
            "ready_seconds": ready_seconds,
            "inference_seconds": inference_wall,
            "usage": usage,
            "command": command,
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_file.close()


def run_qw3_agent(agent, model, prompt, workdir, timeout):
    trace = workdir / "qw3-agent.jsonl"
    store = workdir / "store"
    store.mkdir()
    command = [
        str(agent),
        "--metal",
        "-m", str(model),
        "--ctx", "8192",
        "--no-prefix-cache",
        "--nothink",
        "--temp", "0",
        "--seed", "1",
        "--max-tool-rounds", "4",
        "-n", "256",
        "--trace", str(trace),
        "--store-dir", str(store),
        "-p", prompt,
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=agent.parent,
        text=True,
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    wall = time.monotonic() - started
    if completed.returncode != 0:
        fail("qw3-agent exited unsuccessfully", completed.stderr[-4000:])
    events = []
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    inference = [event for event in events if event.get("event") == "inference"]
    tools = [event.get("name") for event in events if event.get("event") == "tool"]
    return {
        "output": completed.stdout,
        "stderr": completed.stderr,
        "tool_rounds": len(tools),
        "tools": tools,
        "wall_seconds": wall,
        "prefill_seconds": sum(float(e.get("prefill_ms", 0)) for e in inference) / 1000,
        "generation_seconds": sum(float(e.get("generation_ms", 0)) for e in inference) / 1000,
        "prefill_tokens": sum(int(e.get("prefill_tokens", 0)) for e in inference),
        "generation_tokens": sum(int(e.get("generation_tokens", 0)) for e in inference),
        "command": command,
    }


def extract_json_object(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_output(label, text, expected):
    value = extract_json_object(text)
    if value is None:
        return False, f"{label}: no JSON object found"
    if set(value) != set(REFERENCE_FIELDS):
        return False, f"{label}: keys differ: {sorted(value)}"
    if value != expected:
        return False, f"{label}: values differ\nexpected={expected}\nactual={value}"
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "QW3_DENSE_MODEL", "../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf"),
        help="candidate model used by qw3-agent",
    )
    parser.add_argument(
        "--reference-model",
        default=os.environ.get(
            "LLAMA_REFERENCE_MODEL",
            os.environ.get(
                "QW3_DENSE_MODEL",
                "../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf")),
        help="llama.cpp-compatible model used by llama-server",
    )
    parser.add_argument("--agent", default=os.environ.get("QW3_AGENT_BIN", "./qw3-agent"))
    parser.add_argument(
        "--llama-server",
        default=os.environ.get(
            "LLAMA_SERVER", "../../llama.cpp/build/bin/llama-server"),
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="run only qw3-agent against the live oracle",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    model = (repo / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    reference_model = ((repo / args.reference_model).resolve()
                       if not Path(args.reference_model).is_absolute()
                       else Path(args.reference_model))
    agent = (repo / args.agent).resolve() if not Path(args.agent).is_absolute() else Path(args.agent)
    server_arg = Path(args.llama_server)
    if server_arg.is_absolute():
        server = str(server_arg)
    elif len(server_arg.parts) > 1:
        candidate_server = (repo / server_arg).resolve()
        server = str(candidate_server) if candidate_server.is_file() else None
    else:
        server = shutil.which(args.llama_server)
    if not model.is_file():
        fail(f"model not found: {model}")
    if not args.candidate_only and not reference_model.is_file():
        fail(f"reference model not found: {reference_model}")
    if not agent.is_file():
        fail(f"agent not found: {agent}")
    if not args.candidate_only and not server:
        fail(f"llama-server not found: {args.llama_server}")

    with tempfile.TemporaryDirectory(prefix="qw3-agent-reference-") as temp:
        workdir = Path(temp)
        try:
            live = request_json(REFERENCE_URL, timeout=args.timeout)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            fail("cannot fetch the live GitHub reference", str(exc))
        expected = {
            "full_name": live.get("full_name"),
            "default_branch": live.get("default_branch"),
            "license": (live.get("license") or {}).get("spdx_id"),
            "archived": live.get("archived"),
        }
        if any(value is None for value in expected.values()):
            fail("live GitHub reference is missing a required field", json.dumps(live)[:2000])
        prompt = (
            "This is a live-internet agent evaluation. Use visit_url exactly once on "
            f"{REFERENCE_URL}. Read the returned GitHub repository JSON. Return only one "
            "JSON object with exactly these keys: full_name, default_branch, license, "
            "archived. license must be license.spdx_id and archived must be a JSON boolean. "
            "Copy live values verbatim; do not answer from memory and do not add prose."
        )

        reference = None
        if not args.candidate_only:
            reference = run_llama_reference(
                str(server), reference_model, REFERENCE_URL, prompt, workdir,
                args.timeout)
            ref_ok, ref_error = validate_output(
                "llama-server", reference["output"], expected)
            if not ref_ok:
                fail(ref_error, reference["output"])
            if reference["tool_calls"] != 1:
                fail("llama-server reference did not call visit_url exactly once")

        candidate = run_qw3_agent(
            agent, model, prompt, workdir, args.timeout)
        qw3_ok, qw3_error = validate_output(
            "qw3-agent", candidate["output"], expected)
        if not qw3_ok:
            fail(qw3_error, candidate["output"] + "\n" + candidate["stderr"][-2000:])
        if candidate["tools"].count("visit_url") != 1:
            fail("qw3-agent did not call visit_url exactly once", json.dumps(candidate["tools"]))

        summary = {
            "quality": "match",
            "reference_url": REFERENCE_URL,
            "expected": expected,
            "qw3_agent": {
                "model": model.name,
                "tool_rounds": candidate["tool_rounds"],
                "wall_seconds": round(candidate["wall_seconds"], 3),
                "prefill_seconds": round(candidate["prefill_seconds"], 3),
                "generation_seconds": round(candidate["generation_seconds"], 3),
                "prefill_tokens": candidate["prefill_tokens"],
                "generation_tokens": candidate["generation_tokens"],
            },
        }
        if reference is not None:
            summary["llama_server"] = {
                "model": reference_model.name,
                "tool_rounds": reference["tool_rounds"],
                "tool_calls": reference["tool_calls"],
                "ready_seconds": round(reference["ready_seconds"], 3),
                "inference_seconds": round(reference["inference_seconds"], 3),
                **reference["usage"],
            }
        print("test-agent-llama-reference: ok")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
