#!/usr/bin/env python3
"""End-to-end OpenAI API verification for qw3-server with Inkling 2 Mini."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_req(url, method="GET", payload=None, headers=None, timeout=180):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = dict(headers or {})
    if data is not None:
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8")


def sse_req(url, payload, timeout=180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    chunks = []
    saw_done = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        assert resp.status == 200, f"SSE status {resp.status}"
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                saw_done = True
                break
            if line.startswith("data: "):
                chunks.append(json.loads(line[6:]))
    return chunks, saw_done


def main():
    root = Path(__file__).resolve().parent.parent
    model = os.environ.get(
        "QW3_INKLING_MODEL",
        str((root / "../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf").resolve()),
    )
    server_bin = str(root / "qw3-server")
    port = free_port()
    base = f"http://127.0.0.1:{port}"

    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
    proc = subprocess.Popen(
        [server_bin, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--ctx", "32768"],
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # 1. Wait for /health
        healthy = False
        for _ in range(120):
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", "replace")
                raise RuntimeError(f"qw3-server exited early ({proc.returncode}):\n{err}")
            try:
                st, _, body = http_req(f"{base}/health", timeout=2)
                if st == 200:
                    h = json.loads(body)
                    assert h["status"] == "ok", h
                    assert h["ctx_size"] == 32768, h
                    healthy = True
                    print("[1/9] /health OK:", h)
                    break
            except OSError:
                time.sleep(0.25)
        assert healthy, "server did not become healthy"

        # 2. /v1/models & CORS OPTIONS
        st, _, body = http_req(f"{base}/v1/models")
        assert st == 200
        models = json.loads(body)
        assert models["object"] == "list" and len(models["data"]) == 1
        model_id = models["data"][0]["id"]
        print("[2/9] /v1/models OK:", model_id)

        st, hdrs, _ = http_req(f"{base}/v1/chat/completions", method="OPTIONS")
        assert st == 204 and hdrs.get("Access-Control-Allow-Origin") == "*"

        # 3. /tokenize
        st, _, body = http_req(f"{base}/tokenize", "POST", {"content": "Ciao mondo!"})
        assert st == 200
        tok_res = json.loads(body)
        assert tok_res["count"] == len(tok_res["tokens"]) and tok_res["count"] > 0
        print("[3/9] /tokenize OK:", tok_res)

        # 4. Non-streaming chat completion with thinking -> reasoning_content + content
        st, _, body = http_req(
            f"{base}/v1/chat/completions",
            "POST",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Quanto fa 17 * 6? Rispondi solo col numero."}],
                "temperature": 0,
                "max_tokens": 512,
                "reasoning_effort": "medium",
            },
        )
        assert st == 200, body
        r = json.loads(body)
        msg = r["choices"][0]["message"]
        assert "102" in (msg.get("content") or ""), msg
        assert len(msg.get("reasoning_content") or "") > 0, f"expected reasoning_content: {msg}"
        print(
            f"[4/9] chat non-stream (think) OK: content={msg['content']!r} "
            f"reasoning_len={len(msg['reasoning_content'])} "
            f"pp={r['timings']['prompt_per_second']:.1f} tg={r['timings']['predicted_per_second']:.1f} tok/s"
        )

        # 5. Streaming SSE chat completion + include_usage (Turn 1, reasoning_effort=none)
        turn1_messages = [
            {"role": "system", "content": "Rispondi in modo brevissimo."},
            {"role": "user", "content": "Ricorda il codice ALPHA-99 e rispondi solo OK."},
        ]
        chunks, saw_done = sse_req(
            f"{base}/v1/chat/completions",
            {
                "model": model_id,
                "messages": turn1_messages,
                "temperature": 0,
                "max_tokens": 128,
                "reasoning_effort": "none",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert saw_done, "missing [DONE]"
        stream_text = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c.get("choices")
        )
        usage_chunk = [c for c in chunks if c.get("usage")]
        assert len(usage_chunk) == 1, "missing usage chunk"
        assert "OK" in stream_text.upper(), f"unexpected stream text: {stream_text!r}"
        print(
            f"[5/9] chat SSE stream OK: text={stream_text!r} "
            f"tg={usage_chunk[0]['timings']['predicted_per_second']:.1f} tok/s"
        )

        # 6. Multi-turn KV prefix reuse (Turn 2 appends to Turn 1)
        turn2_messages = turn1_messages + [
            {"role": "assistant", "content": stream_text},
            {"role": "user", "content": "Qual era il codice che ti ho chiesto di ricordare? Rispondi solo col codice."},
        ]
        st, _, body = http_req(
            f"{base}/v1/chat/completions",
            "POST",
            {
                "model": model_id,
                "messages": turn2_messages,
                "temperature": 0,
                "max_tokens": 128,
                "reasoning_effort": "none",
            },
        )
        assert st == 200, body
        r2 = json.loads(body)
        ans2 = r2["choices"][0]["message"]["content"]
        cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert "ALPHA-99" in ans2, f"multi-turn recall failed: {ans2!r}"
        assert cached > 0, f"expected KV prefix cache hit > 0, got {r2['usage']}"
        print(
            f"[6/9] multi-turn KV reuse OK: ans={ans2!r} cached={cached}/{r2['usage']['prompt_tokens']}"
        )

        # 7. OpenAI tools / tool_calls + tool response roundtrip
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City name"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        t_msgs = [{"role": "user", "content": "Che tempo fa a Roma? Usa lo strumento get_weather."}]
        st, _, body = http_req(
            f"{base}/v1/chat/completions",
            "POST",
            {
                "model": model_id,
                "messages": t_msgs,
                "tools": tools,
                "temperature": 0,
                "max_tokens": 256,
                "reasoning_effort": "none",
            },
        )
        assert st == 200, body
        rt = json.loads(body)
        choice = rt["choices"][0]
        assert choice["finish_reason"] == "tool_calls", rt
        tcs = choice["message"].get("tool_calls") or []
        assert len(tcs) >= 1 and tcs[0]["function"]["name"] == "get_weather", rt
        args_obj = json.loads(tcs[0]["function"]["arguments"])
        assert "rom" in args_obj.get("city", "").lower(), args_obj
        print("[7a/9] OpenAI tool_call OK:", tcs[0])

        t_msgs2 = t_msgs + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tcs[0]],
            },
            {
                "role": "tool",
                "tool_call_id": tcs[0]["id"],
                "content": json.dumps({"city": "Roma", "temp_c": 22, "condition": "soleggiato"}),
            },
        ]
        st, _, body = http_req(
            f"{base}/v1/chat/completions",
            "POST",
            {
                "model": model_id,
                "messages": t_msgs2,
                "tools": tools,
                "temperature": 0,
                "max_tokens": 256,
                "reasoning_effort": "none",
            },
        )
        assert st == 200, body
        rt2 = json.loads(body)
        final_tool_ans = rt2["choices"][0]["message"]["content"]
        assert "22" in final_tool_ans and "Ġ" not in final_tool_ans, final_tool_ans
        print("[7b/9] OpenAI tool response roundtrip OK:", final_tool_ans)

        # 8. Stop sequence & /v1/completions
        st, _, body = http_req(
            f"{base}/v1/chat/completions",
            "POST",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Scrivi i numeri da 1 a 10 separati da virgola: 1, 2, 3, ..."}],
                "temperature": 0,
                "max_tokens": 128,
                "reasoning_effort": "none",
                "stop": ["5"],
            },
        )
        assert st == 200, body
        rs = json.loads(body)
        stop_txt = rs["choices"][0]["message"]["content"]
        assert rs["choices"][0]["finish_reason"] == "stop" and "5" not in stop_txt, rs
        print("[8/9] stop sequence OK:", repr(stop_txt))

        # 9. Vision image_url guard (Step 2 placeholder) + /v1/completions
        st, _, body = http_req(
            f"{base}/v1/chat/completions",
            "POST",
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Descrivi"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        ],
                    }
                ],
            },
        )
        assert st == 400 and "mmproj" in body, (st, body)
        st, _, body = http_req(
            f"{base}/v1/completions",
            "POST",
            {"model": model_id, "prompt": "The capital of Italy is", "temperature": 0, "max_tokens": 16},
        )
        assert st == 200 and len(json.loads(body)["choices"][0]["text"]) > 0, (st, body)
        print("[9/9] image_url 400 guard & /v1/completions OK")
        print("test-server: ALL 9 CHECKS PASSED")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
