#!/bin/sh
set -eu

DENSE_MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_PROFILE_BIN:-./qw3-cpu-test}"

[ -x "$BIN" ] || {
    echo "test-dense-session: missing $BIN" >&2
    exit 1
}
[ -r "$DENSE_MODEL" ] || {
    echo "test-dense-session: missing $DENSE_MODEL" >&2
    exit 1
}

out=$(QW3_CPU_THREADS="${QW3_CPU_THREADS:-16}" \
      "$BIN" --cpu -m "$DENSE_MODEL" -p c \
      --session-roundtrip --ctx 32 2>/dev/null)

echo "$out" | grep -q \
    'logits=exact decode_steps=4 maxdiff=0 rmsdiff=0 corrupt=rejected' || {
    echo "test-dense-session: roundtrip mismatch: $out" >&2
    exit 1
}

echo "test-dense-session: ok"
