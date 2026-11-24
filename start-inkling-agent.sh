#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
cd "$root"

# --kv-q8 halves Inkling KV cache RAM (~1.15 GiB at 64k ctx vs ~1.16 GiB FP16 at 32k).
# --temp 0.6 is a balanced default for coding; override with e.g. --temp 1.0 (model card),
# --think-effort 0.5, --think-budget 512, or --ctx 131072 via "$@".
exec ./qw3-agent --metal --kv-q8 --ctx 65536 \
    --temp 0.6 --top-p 1.0 --sample-top-k 0 --min-p 0 \
    -m "$root/../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf" \
    --store-dir "$HOME/.qw3-agent-inkling" "$@"
