#!/bin/sh
set -eu

DENSE_MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_PROFILE_BIN:-./qw3-cpu-test}"

[ -x "$BIN" ] || { echo "test-dense-forward: missing $BIN" >&2; exit 1; }
[ -r "$DENSE_MODEL" ] || {
    echo "test-dense-forward: missing $DENSE_MODEL" >&2
    exit 1
}

out=$(QW3_CPU_THREADS="${QW3_CPU_THREADS:-16}" \
      QW3_PROBE_DENSE_FORWARD=1 \
      "$BIN" --cpu -m "$DENSE_MODEL" --probe-token 66 -n 0 -p test \
      2>/dev/null)

echo "$out" | grep -q '^dense_ffn0:.*rms=' || {
    echo "test-dense-forward: dense FFN probe failed" >&2
    exit 1
}
echo "$out" | grep -q '^dense_forward: layers=64 argmax=' || {
    echo "test-dense-forward: 64-layer forward failed" >&2
    exit 1
}

echo "test-dense-forward: ok"
