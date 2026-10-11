#!/usr/bin/env python3
"""Real-tokenizer regression: read a fixture, print formatting, resume the session."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def assistant_answers(transcript):
    return re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|<\|im_start\|>|\Z)",
                      transcript, re.S)


def main():
    model = Path(os.environ.get("QW3_MODEL", "../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")).resolve()
    agent = Path(os.environ.get("QW3_AGENT_BIN", "./qw3-agent")).resolve()
    artifacts = os.environ.get("QW3_REPEAT_ARTIFACT_DIR")
    with tempfile.TemporaryDirectory(prefix="qw3-repeat-format-") as temporary:
        tmp = Path(temporary)
        work, store = tmp / "work", tmp / "store"
        work.mkdir()
        store.mkdir()
        diagram = "┌" + "─" * 80 + "┐\n│ Core -> Tools -> Risposta │\n└" + "─" * 80 + "┘"
        separator = "=" * 120
        (work / "README.md").write_text(
            "# Progetto di prova\nCore coordina i tool e genera la risposta.\n\n"
            "Diagramma da riportare nella risposta:\n" + diagram + "\n\n"
            "Separatore da riportare nella risposta:\n" + separator + "\n")
        env = os.environ.copy()
        env["QW3_METAL_KERNEL_DIR"] = str(ROOT / "metal")
        common = [str(agent), "--metal", "-m", str(model), "--ctx", "8192",
                  "--kv-f16", "--nothink", "--temp", "0", "--no-prefix-cache",
                  "--chdir", str(work), "--store-dir", str(store),
                  "--conversation", "repeat-format", "--max-tool-rounds", "3"]

        def run(name, prompt, tokens):
            result = subprocess.run(common + ["-p", prompt, "-n", str(tokens)],
                                    env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=300)
            output = result.stdout.decode("utf-8", errors="replace")
            (tmp / (name + ".out")).write_text(output)
            if result.returncode:
                raise AssertionError(name + " failed: " + output)
            if "stopped repeated" in output:
                raise AssertionError(name + " hit the repeat guard: " + output)
            return output

        try:
            first = run("after-tool",
                        "Analizza il progetto nella cartella corrente. Prima usa read per leggere "
                        "README.md. Nella risposta finale riporta esattamente il diagramma e il "
                        "separatore contenuti nel README, poi scrivi AGENT_DIAGRAM_OK. "
                        "Non modificare file e non chiamare altri tool.", 512)
            if "[tool] read" not in first:
                raise AssertionError("read was not called: " + first)
            transcript = (store / "repeat-format.txt").read_text()
            (tmp / "after-tool-transcript.txt").write_text(transcript)
            answers = assistant_answers(transcript)
            final = answers[-1] if answers else ""
            # Inspect only the assistant answer: the tool also contains these strings.
            if "AGENT_DIAGRAM_OK" not in final or diagram not in final or separator not in final:
                raise AssertionError("assistant did not finish the fixture: " + final)
            second = run("resume", "Rispondi soltanto con SESSION_FORMAT_OK. Non usare tool.", 32)
            if "KV resumed" not in second:
                raise AssertionError("checkpoint did not resume: " + second)
            answers = assistant_answers((store / "repeat-format.txt").read_text())
            if not answers or "SESSION_FORMAT_OK" not in answers[-1]:
                raise AssertionError("follow-up answer missing: " + second)
            print("test-agent-repeat-format: ok (read, Unicode box, ASCII separator, KV resume)")
        finally:
            if artifacts:
                destination = Path(artifacts)
                destination.mkdir(parents=True, exist_ok=True)
                for path in tmp.glob("*.out"):
                    shutil.copyfile(path, destination / path.name)
                for path in tmp.glob("*-transcript.txt"):
                    shutil.copyfile(path, destination / path.name)
                shutil.copyfile(work / "README.md", destination / "fixture.md")


if __name__ == "__main__":
    main()
