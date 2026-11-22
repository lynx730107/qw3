#!/bin/sh
set -eu
d=$(dirname "$0")
export QW3_METAL_NGL=64 QW3_METAL_KV_F16=1 QW3_METAL_IQ3_NIBBLE=1 QW3_METAL_DENSE_PARTIAL_TILES=1
export QW3_DENSE_MODEL=../../models/Qwen3.8-27B-UD-IQ3_XXS.gguf QW3_MODEL=../../models/Qwen3.8-27B-UD-IQ3_XXS.gguf
make -j4 qw3-bench qw3-agent qw3-cli qw3-agent-context-pressure-test
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
cc -O3 -I. "$d/logits.c" qw3_metal_core.o qw3_metal.o -framework Foundation -framework Metal -lm -lpthread -o "$t/logits-final"
for mode in 0 1; do QW3_METAL_IQ1_BATCH=$mode "$t/logits-final" "$QW3_MODEL" 63 "$t/final-$mode.bin" >"$d/prod-logits-$mode.log" 2>&1; done
python3 "$d/compare.py" "$t/final-0.bin" "$t/final-1.bin" >"$d/prod-compare.log"
for block in 1-old 2-new 3-q3 4-q3 5-new 6-old; do
 model=$QW3_MODEL
 case "$block" in
 *old) export QW3_METAL_IQ1_BATCH=0 ;;
 *new) unset QW3_METAL_IQ1_BATCH ;;
 *q3) unset QW3_METAL_IQ1_BATCH; model=../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf ;;
 esac
 ./qw3-bench --metal --llama-style --ngl 64 --ctx-alloc 1280 -t 4 -m "$model" -p 128 -n 0 -r 3 >"$d/prod-$block.log" 2>&1
 tail -1 "$d/prod-$block.log"
done
unset QW3_METAL_IQ1_BATCH
python3 "$d/italiano.py" "$d" >"$d/italiano.jsonl"
sh tests/test_dense_agent.sh >"$d/checkpoint.log" 2>&1
sh tests/test_agent_coding_smoke.sh >"$d/prod-coding.log" 2>&1
make test-dense-metal-logits >"$d/prod-reference.log" 2>&1
sh tests/test_agent_python_smoke.sh >"$d/prod-python.log" 2>&1
sh tests/test_agent_git_smoke.sh >"$d/prod-git.log" 2>&1
python3 tests/test_agent_llama_reference.py --model "$QW3_MODEL" --candidate-only >"$d/prod-web.log" 2>&1
python3 "$d/italiano.py" "$d" ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf >"$d/italiano-q3.jsonl"
make test-agent-context-pressure test-agent-bash test-agent-edit test-agent-web-page >"$d/context-tools.log" 2>&1
