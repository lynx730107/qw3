#!/usr/bin/env python3
"""End-to-end OpenAI Vision API test suite for qw3-server (--mmproj inkling2v)."""

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_BIN = os.path.join(ROOT, "qw3-server")
MODEL = os.environ.get(
    "QW3_MODEL",
    os.path.abspath(os.path.join(ROOT, "..", "..", "models", "Inkling-2-Mini-UD-Q3_K_XL.gguf")),
)
MMPROJ = os.environ.get(
    "QW3_MMPROJ",
    os.path.abspath(os.path.join(ROOT, "..", "..", "models", "mmproj-Inkling-2-Mini-F16.gguf")),
)
TEST_JPEG = os.path.abspath(os.path.join(ROOT, "..", "..", "llama.cpp", "tools", "mtmd", "test-1.jpeg"))


def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def encode_png(width, height, rgb_bytes):
    """Pure-Python minimal RGB PNG encoder (zero dependencies)."""
    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)  # filter type 0
        raw.extend(rgb_bytes[y * stride:(y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def make_shapes_png():
    """320x200 white canvas with a solid RED square on the left and solid BLUE circle on the right."""
    W, H = 320, 200
    buf = bytearray([255] * (W * H * 3))
    for y in range(H):
        for x in range(W):
            off = (y * W + x) * 3
            # Red square: x in [30..120], y in [55..145]
            if 30 <= x <= 120 and 55 <= y <= 145:
                buf[off:off + 3] = bytes((220, 20, 20))
            # Blue circle: center (235, 100), radius 48
            dx, dy = x - 235, y - 100
            if dx * dx + dy * dy <= 48 * 48:
                buf[off:off + 3] = bytes((20, 60, 230))
    return encode_png(W, H, bytes(buf))


# Crisp 5x7 pixel font for OCR test
FONT_5X7 = {
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "4": ["10010", "10010", "10010", "11111", "00010", "00010", "00010"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
}


def make_ocr_png(text="QW3-42"):
    """Crisp black-on-white banner displaying text at 8x scale."""
    scale = 8
    char_w = 6 * scale
    pad_x, pad_y = 32, 32
    W = pad_x * 2 + len(text) * char_w
    H = pad_y * 2 + 7 * scale
    buf = bytearray([255] * (W * H * 3))
    for ci, ch in enumerate(text):
        glyph = FONT_5X7[ch]
        ox = pad_x + ci * char_w
        for gy in range(7):
            for gx in range(5):
                if glyph[gy][gx] == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            px = ox + gx * scale + sx
                            py = pad_y + gy * scale + sy
                            off = (py * W + px) * 3
                            buf[off:off + 3] = b"\x00\x00\x00"
    return encode_png(W, H, bytes(buf))


def req_json(url, payload=None, timeout=180):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def req_sse(url, payload, timeout=180):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    saw_done = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(body))
    return chunks, saw_done


def main():
    assert os.path.exists(SERVER_BIN), f"missing {SERVER_BIN}"
    assert os.path.exists(MODEL), f"missing {MODEL}"
    assert os.path.exists(MMPROJ), f"missing {MMPROJ}"

    port = pick_free_port()
    base = f"http://127.0.0.1:{port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}

    cmd = [
        SERVER_BIN, "--metal", "--ctx", "16384", "--port", str(port),
        "-m", MODEL, "--mmproj", MMPROJ,
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Wait for health
        t0 = time.time()
        health = None
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"server exited early:\n{err}")
            try:
                _, health = req_json(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(0.25)
        assert health is not None, "server did not become healthy"
        assert health.get("vision") is True, f"expected vision=true: {health}"
        assert health.get("vision_arch") == "inkling2v", f"unexpected arch: {health}"
        print(f"1. /health OK: {health}")

        # 2. Shapes + Colors (base64 PNG)
        shapes_b64 = base64.b64encode(make_shapes_png()).decode("ascii")
        shapes_uri = f"data:image/png;base64,{shapes_b64}"
        _, r1 = req_json(base + "/v1/chat/completions", {
            "model": "inkling",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": shapes_uri}},
                    {"type": "text", "text": "Name the two colored geometric shapes in this image (left and right) concisely."},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 128,
            "reasoning_effort": "none",
        })
        ans1 = r1["choices"][0]["message"]["content"]
        u1 = r1["usage"]
        print(f"2. Shapes/Colors OK: ans={ans1!r} prompt_tok={u1['prompt_tokens']} tg={r1['timings']['predicted_per_second']:.1f} tok/s")
        low1 = ans1.lower()
        assert "red" in low1 and "square" in low1, f"missed red square: {ans1}"
        assert "blue" in low1 and "circle" in low1, f"missed blue circle: {ans1}"

        # 3. Multi-turn follow-up on SAME image -> KV cache + vision cache hit!
        t_turn2 = time.time()
        _, r2 = req_json(base + "/v1/chat/completions", {
            "model": "inkling",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": shapes_uri}},
                        {"type": "text", "text": "Name the two colored geometric shapes in this image (left and right) concisely."},
                    ],
                },
                {"role": "assistant", "content": ans1},
                {"role": "user", "content": "Which color is the circle? Reply with one word."},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
            "reasoning_effort": "none",
        })
        ans2 = r2["choices"][0]["message"]["content"]
        cached2 = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
        pt2 = r2["usage"]["prompt_tokens"]
        print(f"3. Multi-turn Vision KV reuse OK: ans={ans2!r} cached={cached2}/{pt2} ({(time.time()-t_turn2)*1000:.0f} ms total)")
        assert "blue" in ans2.lower(), f"expected blue: {ans2}"
        assert cached2 > 40, f"expected image+turn1 KV cache hit (>40 tokens), got {cached2}"

        # 4. Crisp OCR test ("QW3-42")
        ocr_b64 = base64.b64encode(make_ocr_png("QW3-42")).decode("ascii")
        _, r3 = req_json(base + "/v1/chat/completions", {
            "model": "inkling",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ocr_b64}"}},
                    {"type": "text", "text": "Read the exact text shown in this image. Reply with only the text."},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 384,
            "reasoning_effort": "none",
        })
        ans3 = r3["choices"][0]["message"]["content"]
        print(f"4. Vision OCR OK: ans={ans3!r} prompt_tok={r3['usage']['prompt_tokens']}")
        assert "QW3-42" in ans3.upper(), f"expected QW3-42 in OCR output: {ans3}"

        # 5. Real JPEG via file:// URL + SSE streaming
        if os.path.exists(TEST_JPEG):
            chunks, done = req_sse(base + "/v1/chat/completions", {
                "model": "inkling",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{TEST_JPEG}"}},
                        {"type": "text", "text": "Describe this image in one short sentence."},
                    ],
                }],
                "temperature": 0.0,
                "max_tokens": 256,
                "reasoning_effort": "none",
                "stream": True,
                "stream_options": {"include_usage": True},
            })
            assert done, "missing [DONE] in vision SSE"
            text4 = "".join(
                c["choices"][0]["delta"].get("content", "")
                for c in chunks if c.get("choices")
            )
            usage_chunks = [c for c in chunks if c.get("usage")]
            assert usage_chunks, "missing usage chunk in vision SSE"
            u4 = usage_chunks[-1]["usage"]
            print(f"5. Real JPEG (file://) + SSE OK: ans={text4!r} prompt_tok={u4['prompt_tokens']}")
            assert len(text4.strip()) > 10, f"too short description: {text4}"

        print("ALL VISION TESTS PASSED!")
    finally:
        proc.terminate()
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
        lines = [l for l in stderr.decode("utf-8", errors="replace").splitlines() if "qw3-server:" in l]
        print("\n--- Server log summary ---")
        for l in lines[-10:]:
            print(" ", l)


if __name__ == "__main__":
    main()
