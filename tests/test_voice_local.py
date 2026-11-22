#!/usr/bin/env python3
"""Real Italian TTS -> ASR smoke test, on an isolated ephemeral audio.cpp server.

Requires make voice-setup and ffmpeg for the test WAV resampling. Never records.
QW3_VOICE_TEST_PLAYBACK enables real progressive playback; QW3_VOICE_TEST_MODEL
also adds a real agent turn that speaks.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import wave
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent


def main():
    if not shutil.which("ffmpeg"):
        raise SystemExit("This smoke test needs ffmpeg to resample Kokoro's WAV to 16 kHz.")
    server_bin = ROOT / ".deps/audio.cpp/build/qw3-metal/bin/audiocpp_server"
    server_template = ROOT / "config/audio-cpp.local.json"
    if not server_bin.exists() or not server_template.exists():
        raise SystemExit("Run make voice-setup first.")
    with tempfile.TemporaryDirectory(prefix="qw3-voice-local-") as directory:
        tmp = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server_cfg = json.loads(server_template.read_text())
        server_cfg["port"] = port
        (tmp / "server.json").write_text(json.dumps(server_cfg))
        client_cfg = {"server_url": f"http://127.0.0.1:{port}", "request_timeout_seconds": 180,
                      "tts_speed": 1}
        (tmp / "client.json").write_text(json.dumps(client_cfg))
        with (tmp / "server.log").open("w+") as log:
            server = subprocess.Popen([str(server_bin), "--config", str(tmp / "server.json"), "--no-ui"],
                                      stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            try:
                for _ in range(100):
                    if server.poll() is not None:
                        raise RuntimeError("audio.cpp exited during startup")
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                            if json.load(response)["status"] == "ok":
                                break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("audio.cpp startup timed out")

                def run(*args):
                    result = subprocess.run(list(map(str, args)), capture_output=True, text=True,
                                            timeout=240, cwd=ROOT)
                    if result.returncode:
                        raise RuntimeError(result.stdout + result.stderr)
                    return result

                client = [ROOT / "qw3-voice", "--config", tmp / "client.json"]
                run(*client, "check")
                before = time.monotonic()
                run(*client, "speak", "Ciao, questa è una prova della voce italiana.", tmp / "speech.wav")
                print(f"Kokoro Italian WAV generated in {time.monotonic() - before:.2f} s (cold load included)", flush=True)
                run("ffmpeg", "-v", "error", "-y", "-i", tmp / "speech.wav", "-ar", "16000", "-ac", "1", tmp / "input.wav")
                client_cfg["tts_speed"] = 0.75
                (tmp / "client.json").write_text(json.dumps(client_cfg))
                run(*client, "speak", "Ciao, questa è una prova della voce italiana.", tmp / "slow.wav")
                run("ffmpeg", "-v", "error", "-y", "-i", tmp / "slow.wav", "-ar", "16000", "-ac", "1", tmp / "slow-input.wav")
                def duration(path):
                    with wave.open(str(path), "rb") as audio:
                        return audio.getnframes() / audio.getframerate()
                normal = duration(tmp / "input.wav")
                slow = duration(tmp / "slow-input.wav")
                if slow <= normal * 1.1:
                    raise RuntimeError(f"Kokoro did not slow down: normal={normal:.2f}s, slow={slow:.2f}s")
                print(f"Kokoro request speed verified: {normal:.2f} s at 1.0, {slow:.2f} s at 0.75", flush=True)
                client_cfg["tts_speed"] = 1
                (tmp / "client.json").write_text(json.dumps(client_cfg))
                before = time.monotonic()
                transcript = run(*client, "transcribe", tmp / "input.wav").stdout.strip()
                print(f"Qwen3-ASR transcript in {time.monotonic() - before:.2f} s: {transcript}", flush=True)
                if "prova" not in transcript.lower() or "italiana" not in transcript.lower():
                    raise RuntimeError("Italian roundtrip did not recover the expected words")
                if os.environ.get("QW3_VOICE_TEST_PLAYBACK") or os.environ.get("QW3_VOICE_TEST_MODEL"):
                    client_cfg["speech_chunk_chars"] = 80
                    (tmp / "client.json").write_text(json.dumps(client_cfg))
                    reply = ("La prima misura è ventuno virgola cinque gradi Celsius. "
                             "La seconda misura è diciannove gradi Celsius.")
                    stats = json.loads(run(*client, "say", reply).stdout)
                    if stats["chunks_total"] < 2 or stats["chunks_played"] != stats["chunks_total"]:
                        raise RuntimeError(f"Progressive native playback incomplete: {stats}")
                    if not 0 < stats["first_chunk_ready_ms"] < stats["total_ms"]:
                        raise RuntimeError(f"Invalid progressive timing: {stats}")
                    print("Real Kokoro progressive playback: " + json.dumps(stats), flush=True)
                    client_cfg.pop("speech_chunk_chars")
                    (tmp / "client.json").write_text(json.dumps(client_cfg))
                if os.environ.get("QW3_VOICE_TEST_MODEL"):
                    result = run(ROOT / "qw3-agent-voice", "--voice-config", tmp / "client.json",
                                 "-m", Path(os.environ["QW3_VOICE_TEST_MODEL"]).resolve(), "--ctx", "2048",
                                 "--no-tools", "--nothink", "--temp", "0", "--no-prefix-cache",
                                 "--store-dir", tmp / "store", "-n", "32", "-p", "Rispondi soltanto: Ciao.")
                    statuses = [line for line in result.stderr.splitlines() if line.startswith("voice:")]
                    if statuses != ["voice: synthesizing final answer..."]:
                        raise RuntimeError(result.stderr)
                    print("Real 35B agent -> Kokoro -> native playback: OK", flush=True)
                print("Local Italian speech roundtrip: OK (microphone not used)")
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-6000:])
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
