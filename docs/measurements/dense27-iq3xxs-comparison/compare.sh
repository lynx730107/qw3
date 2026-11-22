#!/bin/sh
# Run from the qw3 root with verified GGUFs available in ../../models.
set -eu
d=docs/measurements/dense27-iq3xxs-comparison
bin=${LLAMA_BIN_DIR:-"$HOME/Downloads/llama.cpp/build/bin"}
a=../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf
b=../../models/Qwen3.8-27B-UD-IQ3_XXS.gguf
export QW3_METAL_IQ3_NIBBLE=1 QW3_METAL_DENSE_PARTIAL_TILES=1 QW3_METAL_KV_F16=1
./qw3-bench --llama-style -m "$b" --metal --ngl 64 -t 4 --ctx-alloc 1280 -p 0 -n 128 -d 1024 -r 1 > "$d/native-IQ3.log" 2>&1 || echo "native exit $?" >> "$d/native-IQ3.log"
for block in A1 B1 B2 A2; do
    case "$block" in A*) model=$a ;; B*) model=$b ;; esac
    "$bin/llama-bench" -m "$model" -ngl 99 -fa on -ctk f16 -ctv f16 -b 128 -ub 128 -t 4 -d 1024 -p 0 -n 128 -r 2 -o json > "$d/llama-$block.json" 2>"$d/llama-$block.log"
done
corpus=$(mktemp)
trap 'rm -f "$corpus"' EXIT
git show 74da025:README.md > "$corpus"
"$bin/llama-perplexity" -m "$b" -f "$corpus" -ngl 99 -fa on -t 4 -c 512 -b 512 -ub 128 --chunks 4 > "$d/ppl-B.log" 2>&1
python3 "$d/reference-tasks.py" "$bin/llama-server" "$b" "$d/tasks-B-server.log" > "$d/tasks-B.jsonl" 2>"$d/tasks-B.stderr"
python3 "$d/reference-web.py" "$bin/llama-server" "$b" "$d/web-B" > "$d/web-B.json" 2>"$d/web-B.stderr"
