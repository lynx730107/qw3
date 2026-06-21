#!/usr/bin/env python3
"""Download small code datasets and render prompts for QW3 expert profiling."""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import textwrap
import urllib.request

HUMANEVAL_X_CPP_URL = (
    "https://raw.githubusercontent.com/zai-org/CodeGeeX/main/"
    "codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download/render code prompts for QW3 SSD hotlist profiling."
    )
    p.add_argument(
        "--dataset",
        choices=("humaneval-x-cpp",),
        default="humaneval-x-cpp",
        help="Dataset to download/render",
    )
    p.add_argument(
        "--out-dir",
        default="datasets/humaneval-x-cpp",
        help="Directory for downloaded and rendered files",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=20,
        help="Maximum tasks to include in the profiling prompt",
    )
    p.add_argument(
        "--mode",
        choices=("c-audit", "c-implement", "mixed"),
        default="mixed",
        help="Prompt style to render",
    )
    return p.parse_args()


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response:
        tmp.write_bytes(response.read())
    tmp.replace(dest)


def read_jsonl_gz(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_cpp_prompt(prompt: str) -> str:
    return prompt.strip()


def render_task(task: dict, mode: str) -> str:
    task_id = task.get("task_id", "unknown")
    prompt = clean_cpp_prompt(str(task.get("prompt", "")))
    example_test = str(task.get("example_test", "")).strip()
    test = str(task.get("test", "")).strip()

    audit_instruction = textwrap.dedent(
        """\
        Analizza il seguente task C/C++ come se stessi facendo code review:
        - identifica edge case, overflow, gestione memoria e complessita';
        - spiega quali invarianti devono essere rispettati;
        - se vedi rischi di sicurezza o undefined behavior, segnalarli.
        """
    ).strip()
    implement_instruction = textwrap.dedent(
        """\
        Completa la funzione in stile C/C++ semplice:
        - preferisci cicli espliciti e controllo dei limiti;
        - evita astrazioni inutili;
        - mantieni compatibilita' con i test forniti.
        """
    ).strip()

    if mode == "c-audit":
        instruction = audit_instruction
    elif mode == "c-implement":
        instruction = implement_instruction
    else:
        instruction = audit_instruction + "\n\n" + implement_instruction

    parts = [
        f"### TASK {task_id}",
        instruction,
        "",
        "```cpp",
        prompt,
        "```",
    ]
    if example_test:
        parts += ["", "Esempi/test minimi:", "```cpp", example_test, "```"]
    if test:
        parts += ["", "Test di riferimento:", "```cpp", test[:3000], "```"]
    parts += [
        "",
        "Rispondi con una breve analisi e poi con una possibile implementazione robusta.",
    ]
    return "\n".join(parts)


def render_dataset(rows: list[dict], out_dir: pathlib.Path, max_tasks: int, mode: str) -> None:
    if max_tasks < 1:
        raise SystemExit("--max-tasks must be positive")
    selected = rows[:max_tasks]
    rendered = [render_task(row, mode) for row in selected]

    prompt_path = out_dir / f"{mode}_profile_prompt.txt"
    jsonl_path = out_dir / f"{mode}_profile_prompts.jsonl"
    manifest_path = out_dir / "manifest.json"

    header = textwrap.dedent(
        f"""\
        Sei un assistente esperto di C e C++. Questo prompt serve a profilare
        gli expert routing del modello su workload di programmazione C/C++.
        Leggi i task seguenti e preparati a rispondere in modo tecnico.

        """
    )
    prompt_path.write_text(header + "\n\n".join(rendered) + "\n", encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fp:
        for row, text in zip(selected, rendered):
            fp.write(json.dumps({"task_id": row.get("task_id"), "prompt": text}) + "\n")

    manifest = {
        "dataset": "humaneval-x-cpp",
        "source_url": HUMANEVAL_X_CPP_URL,
        "mode": mode,
        "tasks_total": len(rows),
        "tasks_rendered": len(selected),
        "prompt_file": str(prompt_path),
        "jsonl_file": str(jsonl_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "humaneval-x-cpp":
        raw_path = out_dir / "humaneval_cpp.jsonl.gz"
        download(HUMANEVAL_X_CPP_URL, raw_path)
        rows = read_jsonl_gz(raw_path)
        render_dataset(rows, out_dir, args.max_tasks, args.mode)
        print(
            f"downloaded/rendered {args.dataset}: {min(args.max_tasks, len(rows))}/"
            f"{len(rows)} tasks in {out_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
