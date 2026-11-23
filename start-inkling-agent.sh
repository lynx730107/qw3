#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
cd "$root"

exec ./qw3-agent --metal --ctx 32768 \
    --temp 0.6 --top-p 1.0 --sample-top-k 0 --min-p 0 \
    -m "$root/../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf" \
    --store-dir "$HOME/.qw3-agent-inkling" "$@"
