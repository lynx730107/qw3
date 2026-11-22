#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"

if [ ! -x ./qw3-server ]; then
    make -j4 qw3-server
fi

MMPROJ="$root/../../models/mmproj-Inkling-2-Mini-F16.gguf"
if [ -f "$MMPROJ" ]; then
    exec ./qw3-server --metal --ctx 32768 --port 8080 \
        -m "$root/../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf" \
        --mmproj "$MMPROJ" "$@"
else
    exec ./qw3-server --metal --ctx 32768 --port 8080 \
        -m "$root/../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf" "$@"
fi
