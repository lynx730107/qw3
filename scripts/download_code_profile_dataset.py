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
        choices=("humaneval-x-cpp", "synthetic-c"),
        default="humaneval-x-cpp",
        help="Dataset to download/render",
    )
    p.add_argument(
        "--out-dir",
        default=None,
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
    language = str(task.get("language", "cpp"))

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
        f"```{language}",
        prompt,
        "```",
    ]
    if example_test:
        parts += ["", "Esempi/test minimi:", f"```{language}", example_test, "```"]
    if test:
        parts += ["", "Test di riferimento:", f"```{language}", test[:3000], "```"]
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
        "dataset": rows[0].get("dataset", "unknown") if rows else "unknown",
        "source_url": rows[0].get("source_url", "local") if rows else "local",
        "mode": mode,
        "tasks_total": len(rows),
        "tasks_rendered": len(selected),
        "prompt_file": str(prompt_path),
        "jsonl_file": str(jsonl_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def synthetic_c_rows() -> list[dict]:
    tasks = [
        (
            "c_parse_u32",
            "Scrivi int parse_u32(const char *s, uint32_t *out) che accetta solo cifre decimali, rifiuta overflow e lascia *out invariato su errore.",
            "assert(parse_u32(\"4294967295\", &v)); assert(!parse_u32(\"4294967296\", &v));",
        ),
        (
            "c_arena_alloc",
            "Implementa un arena allocator C con allineamento power-of-two, controllo overflow su offset+size e funzione reset.",
            "arena_init(&a, buf, sizeof buf); p = arena_alloc(&a, 24, 8); assert(((uintptr_t)p & 7) == 0);",
        ),
        (
            "c_ring_buffer",
            "Completa un ring buffer byte-oriented con push/pop bulk, wrap-around e distinzione pieno/vuoto senza allocazioni dinamiche.",
            "assert(rb_write(&rb, data, 7) == 7); assert(rb_read(&rb, out, 3) == 3);",
        ),
        (
            "c_utf8_scan",
            "Scrivi una funzione C che valida UTF-8, conta codepoint e rifiuta sequenze overlong o surrogate.",
            "assert(utf8_count(\"ciao\", &n) && n == 4); assert(!utf8_count(\"\\xff\", &n));",
        ),
        (
            "c_path_join",
            "Implementa path_join(char *dst, size_t cap, const char *a, const char *b) evitando doppio slash, overflow e buffer non terminati.",
            "assert(path_join(buf, sizeof buf, \"/tmp/\", \"x\") == 0); assert(strcmp(buf, \"/tmp/x\") == 0);",
        ),
        (
            "c_hash_table",
            "Completa una hash table open-addressing per stringhe non owning, con tombstone, resize esplicito e lookup stabile.",
            "assert(ht_put(&h, \"alpha\", 1) == 0); assert(ht_get(&h, \"alpha\", &v) && v == 1);",
        ),
        (
            "c_iovec_builder",
            "Scrivi un builder di iovec che concatena segmenti costanti, controlla il numero massimo di segmenti e calcola la lunghezza totale con overflow check.",
            "assert(iov_add(&b, p, n) == 0); assert(b.total >= n);",
        ),
        (
            "c_tokenizer",
            "Implementa un tokenizer C per config key=value: ignora spazi, supporta commenti #, rifiuta chiavi duplicate e quote non chiuse.",
            "assert(parse_config(\"a=1\\n#x\\nb=two\", &cfg) == 0);",
        ),
        (
            "c_sort_stable",
            "Scrivi merge sort stabile per array di struct item { uint32_t key; uint32_t original; } usando un buffer temporaneo fornito dal chiamante.",
            "items ordinati per key devono mantenere ordine relativo di original a parita' di key.",
        ),
        (
            "c_bitset",
            "Implementa un bitset dinamico su memoria fornita: set, clear, test, find_first_zero, con controlli bounds e nessuna malloc.",
            "assert(bitset_set(&bs, 63) == 0); assert(bitset_test(&bs, 63));",
        ),
        (
            "c_binary_protocol",
            "Completa parser C per header binario little-endian: magic, version, length, crc32; rifiuta pacchetti tronchi e length eccessivo.",
            "assert(parse_packet(buf, len, &pkt) == 0 || parse_packet(buf, len, &pkt) == -1);",
        ),
        (
            "c_lru_cache",
            "Implementa una piccola cache LRU a capacita' fissa con array di nodi, lista doppiamente collegata e mappa key->slot fornita.",
            "get deve promuovere il nodo a MRU; put deve evincere LRU quando piena.",
        ),
    ]
    rows: list[dict] = []
    for task_id, prompt, test in tasks:
        rows.append(
            {
                "dataset": "synthetic-c",
                "source_url": "local",
                "task_id": task_id,
                "language": "c",
                "prompt": prompt,
                "example_test": "",
                "test": test,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir_s = args.out_dir
    if out_dir_s is None:
        out_dir_s = "datasets/synthetic-c" if args.dataset == "synthetic-c" else "datasets/humaneval-x-cpp"
    out_dir = pathlib.Path(out_dir_s)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "humaneval-x-cpp":
        raw_path = out_dir / "humaneval_cpp.jsonl.gz"
        download(HUMANEVAL_X_CPP_URL, raw_path)
        rows = read_jsonl_gz(raw_path)
        for row in rows:
            row["dataset"] = args.dataset
            row["source_url"] = HUMANEVAL_X_CPP_URL
            row.setdefault("language", "cpp")
        render_dataset(rows, out_dir, args.max_tasks, args.mode)
        print(
            f"downloaded/rendered {args.dataset}: {min(args.max_tasks, len(rows))}/"
            f"{len(rows)} tasks in {out_dir}"
        )
    elif args.dataset == "synthetic-c":
        rows = synthetic_c_rows()
        render_dataset(rows, out_dir, args.max_tasks, args.mode)
        print(
            f"rendered {args.dataset}: {min(args.max_tasks, len(rows))}/"
            f"{len(rows)} tasks in {out_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
