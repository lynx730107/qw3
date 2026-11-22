#!/bin/sh
set -eu
out=$(dirname "$0")
export QW3_METAL_NGL=64 QW3_METAL_KV_F16=1 QW3_METAL_IQ3_NIBBLE=1 QW3_METAL_DENSE_PARTIAL_TILES=1
for block in 1-Q3_K_XL 2-IQ3_XXS 3-IQ3_XXS 4-Q3_K_XL; do
    sleep 20
    model="../../models/Qwen3.8-27B-UD-${block#*-}.gguf"
    for phase in pp tg; do
        case "$phase" in
            pp) set -- -p 128 -n 0 -d 0 ;;
            tg) set -- -p 0 -n 128 -d 1024 ;;
        esac
        ./qw3-bench --metal --llama-style --ngl 64 -t 4 --seed 1 --ctx-alloc 1280 -m "$model" "$@" -r 2 > "$out/$block-$phase.log" 2>&1
        tail -1 "$out/$block-$phase.log"
    done
done
