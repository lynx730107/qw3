#!/usr/bin/env python3
"""Generate qw3_streaming_hotlist.inc from QW3 expert profile TSV files."""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import sys

QW3_N_LAYER = 40
QW3_N_EXPERT = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a compiled QW3 SSD streaming hotlist from profile TSVs."
    )
    p.add_argument("profiles", nargs="+", help="Profile TSV files: layer expert hits")
    p.add_argument(
        "-o",
        "--output",
        default="qw3_streaming_hotlist.inc",
        help="Output .inc path",
    )
    p.add_argument(
        "--top",
        type=int,
        default=4096,
        help="Maximum number of (layer, expert) pairs to emit",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=1,
        help="Drop pairs with fewer than this many aggregated hits",
    )
    return p.parse_args()


def read_profiles(paths: list[str]) -> dict[tuple[int, int], int]:
    hits: dict[tuple[int, int], int] = {}
    for path_s in paths:
        path = pathlib.Path(path_s)
        with path.open("r", encoding="utf-8") as fp:
            for lineno, line in enumerate(fp, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    raise SystemExit(f"{path}:{lineno}: expected: layer expert hits")
                try:
                    layer = int(parts[0])
                    expert = int(parts[1])
                    count = int(parts[2])
                except ValueError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid integer field") from exc
                if not (0 <= layer < QW3_N_LAYER):
                    raise SystemExit(f"{path}:{lineno}: layer out of range: {layer}")
                if not (0 <= expert < QW3_N_EXPERT):
                    raise SystemExit(f"{path}:{lineno}: expert out of range: {expert}")
                if count <= 0:
                    continue
                hits[(layer, expert)] = hits.get((layer, expert), 0) + count
    return hits


def write_inc(
    output: str,
    profiles: list[str],
    ordered: list[tuple[tuple[int, int], int]],
    min_hits: int,
    top: int,
) -> None:
    out = pathlib.Path(output)
    source_list = ", ".join(profiles)
    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with out.open("w", encoding="utf-8", newline="\n") as fp:
        fp.write("/* Generated/profiled QW3 SSD streaming hotlist.\n")
        fp.write(" *\n")
        fp.write(" * Format: { layer, expert }, ordered from hottest to coldest.\n")
        fp.write(f" * Generated: {generated}\n")
        fp.write(f" * Sources: {source_list}\n")
        fp.write(f" * Selection: top={top} min_hits={min_hits}\n")
        fp.write(" */\n")
        if ordered:
            fp.write("static const uint16_t qw3_default_streaming_hotlist[][2] = {\n")
            for (layer, expert), count in ordered:
                fp.write(f"    {{ {layer:2d}, {expert:3d} }}, /* hits={count} */\n")
            fp.write("};\n\n")
            fp.write("static const uint32_t qw3_default_streaming_hotlist_count =\n")
            fp.write("    (uint32_t)(sizeof(qw3_default_streaming_hotlist) /\n")
            fp.write("               sizeof(qw3_default_streaming_hotlist[0]));\n")
        else:
            fp.write("static const uint16_t qw3_default_streaming_hotlist[1][2] = {\n")
            fp.write("    { 0, 0 },\n")
            fp.write("};\n\n")
            fp.write("static const uint32_t qw3_default_streaming_hotlist_count = 0;\n")


def main() -> int:
    args = parse_args()
    if args.top < 0:
        raise SystemExit("--top must be non-negative")
    if args.min_hits < 1:
        raise SystemExit("--min-hits must be positive")

    hits = read_profiles(args.profiles)
    ordered = [
        (pair, count)
        for pair, count in sorted(
            hits.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )
        if count >= args.min_hits
    ]
    if args.top:
        ordered = ordered[: args.top]
    else:
        ordered = []

    write_inc(args.output, args.profiles, ordered, args.min_hits, args.top)
    print(
        f"wrote {args.output}: {len(ordered)} entries "
        f"from {len(hits)} unique profiled pairs",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
