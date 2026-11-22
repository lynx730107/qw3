#!/usr/bin/env python3
"""Install the two speech packages and create local configs without overwriting."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".deps" / "audio.cpp"
MODELS = ROOT / ".deps" / "audio-models"


def create_config(path, data):
    try:
        with path.open("x") as output:
            json.dump(data, output, indent=2)
            output.write("\n")
        print(f"Created {path}", flush=True)
    except FileExistsError:
        print(f"Kept existing {path}", flush=True)


def main():
    manager = SOURCE / "tools" / "model_manager_v2.py"
    if not manager.exists():
        raise SystemExit("Run make audio-server first.")
    for package in ["qwen3_asr_0_6b_q8_0", "kokoro_82m_q8_0"]:
        subprocess.run([sys.executable, str(manager), "install", package,
                        "--models-root", str(MODELS)], check=True, cwd=SOURCE)
    server = json.loads((ROOT / "config" / "audio-cpp.example.json").read_text())
    files = [MODELS / "Qwen3-ASR-0.6B-GGUF" / "qwen3-asr-0.6b-q8_0.gguf",
             MODELS / "Kokoro-82M-GGUF" / "kokoro-82m-q8_0.gguf"]
    for entry, path in zip(server["models"], files):
        if not path.is_file():
            raise SystemExit(f"Model package did not install expected file: {path}")
        entry["path"] = str(path)
    create_config(ROOT / "config" / "audio-cpp.local.json", server)
    client = json.loads((ROOT / "config" / "voice.example.json").read_text())
    create_config(ROOT / "config" / "voice.local.json", client)


if __name__ == "__main__":
    main()
